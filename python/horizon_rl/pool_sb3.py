"""Training against the opponent pool with Stable-Baselines3
(docs/rl/design.md#the-opponent-pool).

    env = HorizonVecEnv(server=..., n=16, pairs=True, rounds=True, flat=True, ...)
    pool = Pool(PoolConfig.load("pool.toml"))
    venv = OpponentPoolVecEnv(HorizonSB3VecEnv(env), pool, opponent_device="cpu")
    model = PPO("MlpPolicy", VecMonitor(venv))
    model.learn(10_000_000, callback=[ReleaseKeys(), PoolSnapshots(venv, "runs/sword_a"),
                                      PoolStats(venv, "runs/sword_a")])

Agents 2k and 2k+1 fight each other. Agent 2k is the learner's seat (it keeps it: the arena is
symmetrical), 2k+1 an opponent played by a frozen version from the pool. SB3 sees N/2 envs, the
learner's rows only. Each pair draws its own opponent, in proportion to the weights, whenever
the learner's episode ends (a round with `rounds=True`), and that version plays the pair's whole
next episode. Each step runs one batched, sampled forward pass per version in play.

Every version is loaded once, on `opponent_device`; a version dropped from the pool stays loaded
until no pair plays it any more.
"""
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from .pool import RESULTS, Stats, check_spaces


def round_result(info):
    """The learner's result of the round its `info` ends, or None (no round result: a fault, or
    a run without rounds)."""
    for result in RESULTS:
        if info.get("round_" + result, 0) > 0:
            return result
    return None


def _rows(obs, rows):
    return {k: v[rows] for k, v in obs.items()} if isinstance(obs, dict) else obs[rows]


class OpponentPoolVecEnv(VecEnv):
    """A `HorizonSB3VecEnv` of N agents in pairs, as an SB3 `VecEnv` of its N/2 learner seats.

    `load(path)` returns a model with SB3's `predict(obs, deterministic=False)` and its
    `observation_space` and `action_space`; by default `algo.load(path, device=opponent_device)`.
    """

    def __init__(self, venv, pool, opponent_device="cpu", algo=None, load=None):
        config = getattr(getattr(venv, "env", None), "config", None)
        if config is not None and not config.pairs:
            raise ValueError("the pool needs agents in pairs (pairs=True)")
        if venv.num_envs % 2:
            raise ValueError(f"the pool needs an even number of agents, not {venv.num_envs}")
        if load is None:
            if algo is None:
                from stable_baselines3 import PPO as algo
            load = lambda path: algo.load(path, device=opponent_device)  # noqa: E731
        self.venv, self.pool, self._load = venv, pool, load
        self.models = {}  # version id -> loaded model
        self.stats = Stats()
        pairs = venv.num_envs // 2
        self.playing = [None] * pairs  # each pair's opponent version
        self.rounds_done = 0
        self._opponent_obs = None
        self._actions = None
        super().__init__(pairs, venv.observation_space, venv.action_space)
        for v in list(pool):
            self._ensure_loaded(v)

    # The pool's versions -------------------------------------------------------------------

    def _ensure_loaded(self, version):
        if version.id not in self.models:
            model = self._load(version.path)
            check_spaces(version.name, model.observation_space, model.action_space,
                         self.observation_space, self.action_space)
            self.models[version.id] = model
        return self.models[version.id]

    def _unload_unused(self):
        keep = {v.id for v in self.pool} | {v.id for v in self.playing if v is not None}
        for vid in [vid for vid in self.models if vid not in keep]:
            del self.models[vid]

    def _draw(self, pair):
        self.playing[pair] = v = self.pool.draw()
        self._ensure_loaded(v)
        self._unload_unused()

    def add_snapshot(self, model, path, step):
        """Saves the learner `model` at `path` and adds it to the pool as the snapshot of `step`,
        loaded from the file: a frozen copy. Returns the version dropped, if one was."""
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        dropped = self.pool.add(path, step)
        self._ensure_loaded(self.pool.versions[-1])
        self._unload_unused()
        return dropped

    # VecEnv --------------------------------------------------------------------------------

    def _split(self, obs):
        self._opponent_obs = _rows(obs, slice(1, None, 2))
        return _rows(obs, slice(0, None, 2))

    def reset(self):
        obs = self.venv.reset()
        self.reset_infos = self.venv.reset_infos[0::2]
        return self._split(obs)

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def _opponent_actions(self):
        for pair in range(self.num_envs):
            if self.playing[pair] is None:
                self._draw(pair)
        out = np.zeros_like(self._actions)
        by_version = {}
        for pair, v in enumerate(self.playing):
            by_version.setdefault(v.id, []).append(pair)
        for vid, pairs in by_version.items():
            actions, _ = self.models[vid].predict(_rows(self._opponent_obs, pairs), deterministic=False)
            out[pairs] = actions
        return out

    def step_wait(self):
        full = np.empty((self.venv.num_envs, *self._actions.shape[1:]), self._actions.dtype)
        full[0::2] = self._actions
        full[1::2] = self._opponent_actions()
        obs, rewards, dones, infos = self.venv.step(full)
        obs = self._split(obs)
        infos = infos[0::2]
        for pair in np.flatnonzero(dones[0::2]):
            v = self.playing[pair]
            infos[pair]["opponent_version"] = v.name
            result = round_result(infos[pair])
            if result is not None:
                self.stats.record(v, result)
                self.rounds_done += 1
            self._draw(pair)
        return obs, rewards[0::2], dones[0::2], infos

    def close(self):
        self.venv.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.venv, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.venv, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        # One env behind every index: called once (`release` lets go of all N agents' keys).
        result = self.venv.env_method(method_name, *args, indices=[0], **kwargs)[0]
        return [result for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


def _pool_env(training_env):
    env = training_env
    while not isinstance(env, OpponentPoolVecEnv):
        env = getattr(env, "venv", None)
        if env is None:
            raise TypeError("the training env has no OpponentPoolVecEnv")
    return env


class PoolSnapshots(BaseCallback):
    """Snapshots the learner into the pool every `snapshot_every` learner steps (the pool file),
    as `<run_dir>/snapshots/s<step>.zip`. With nothing listed in the pool file, the first is the
    initial model, at the start."""

    def __init__(self, run_dir, verbose=1):
        super().__init__(verbose)
        self.run_dir = run_dir

    def _snapshot(self, env):
        step = self.model.num_timesteps
        path = env.pool.snapshot_path(self.run_dir, step)
        dropped = env.add_snapshot(self.model, path, step)
        if self.verbose:
            print(f"pool: snapshot {path} ({len(env.pool)} versions"
                  + (f", dropped {dropped.name})" if dropped else ")"), flush=True)

    def _on_training_start(self):
        env = _pool_env(self.training_env)
        if env.pool.needs_initial:
            self._snapshot(env)

    def _on_step(self):
        env = _pool_env(self.training_env)
        if env.pool.due(self.model.num_timesteps):
            self._snapshot(env)
        return True


class PoolStats(BaseCallback):
    """The learner's results against each version, to TensorBoard (`pool/<version>/...`, out of
    the console) and to `<run_dir>/pool_stats.csv`, at the end of every rollout and of training."""

    def __init__(self, run_dir, verbose=0):
        super().__init__(verbose)
        self.path = f"{run_dir}/pool_stats.csv"

    def _on_step(self):
        return True

    def _write(self):
        env = _pool_env(self.training_env)
        for row in env.stats.rows.values():
            rounds = row["rounds"]
            for key in ("rounds", "wins", "losses", "draws", "timeouts"):
                self.logger.record(f"pool/{row['version']}/{key}", row[key], exclude=("stdout", "log"))
            self.logger.record(f"pool/{row['version']}/win_rate", row["wins"] / rounds, exclude=("stdout", "log"))
        self.logger.record("pool/size", len(env.pool))
        env.stats.write_csv(self.path)

    def _on_rollout_end(self):
        self._write()

    def _on_training_end(self):
        self._write()
