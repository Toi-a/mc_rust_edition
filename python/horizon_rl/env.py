"""Gymnasium environments over a `Fleet` (docs/rl/README.md#gymnasium).

- `HorizonVecEnv`: N agents stepped together, each an independent player on the server.
- `HorizonEnv`: one agent.

Observations follow `FleetConfig.obs`: `"state"` a float32 vector, `"image"` an (H, W) uint8
frame, `"both"` a dict of the two. The state is the game's whole fair-play vector
(`env.state_fields`: name -> slice), or with `observation=` the features you list, normalised
unless `normalize=False` (`horizon_rl.observations`; `env.observation_fields` says where each is). Actions: `horizon_rl.actions.ActionCodec` (buttons, hotbar, camera). Rewards: 0 unless a
`reward` function is given (`horizon_rl.rewards`); every signal is in `info`.

The server runs in real time, so nothing here blocks on one agent: an agent whose episode ended
(a death, or a fault: kick, crash, hang) is reset by its supervisor while the others play on.
"""
import dataclasses

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from .actions import BUTTON_NAMES, DEFAULT_BINS, DEFAULT_MAX_TURN, ActionCodec
from .fleet import Fleet, FleetConfig
from .observations import ObservationView

# The action settings of the envs (`ActionCodec`'s), and their defaults.
_CODEC_DEFAULTS = {"camera": "absolute", "buttons": BUTTON_NAMES, "hotbar": True, "flat": False,
                   "delta_max": None, "bins": DEFAULT_BINS, "max_turn": DEFAULT_MAX_TURN, "max_cps": 0.0}


def _encode(codec, actions, states, fields):
    """`actions` for the game, turned from where the `states` (n, L) say each agent looks."""
    aim = None
    if codec.camera == "aim":
        # The turn to the opponent is part of the observation; none while there is no opponent.
        aim = states[:, fields["target_bearing"]] * (states[:, fields["target_present"]] > 0.5)
    return codec.encode(actions, states[:, fields["self_yaw"].start], states[:, fields["self_pitch"].start], aim)


def _open(config, fleet, n, settings):
    """The started fleet of an env, its action settings and its observation view (None: the
    whole state vector as the game gives it)."""
    codec = {k: settings.pop(k, v) for k, v in _CODEC_DEFAULTS.items()}
    observation, normalize = settings.pop("observation", None), settings.pop("normalize", True)
    view = None if observation is None else ObservationView(observation, normalize)
    if view is not None and view.needs_target_health:
        # Asking for the opponent's health is saying the server shows it.
        if fleet is None:
            settings["target_health"] = True
        elif not fleet.config.target_health:
            raise ValueError("the observation has the opponent's health: start the fleet with target_health=True")
    if fleet is None:
        if config is None:
            config = FleetConfig(**settings)
        elif settings:
            config = dataclasses.replace(config, **settings)
        if n is not None and config.n != n:
            config = dataclasses.replace(config, n=n)
        fleet = Fleet(config)
    elif settings:
        raise TypeError(f"settings given with a fleet: {', '.join(settings)}")
    try:
        fleet.start()
        if view is not None:
            view.bind(fleet.state_fields)
    except BaseException:
        fleet.close()
        raise
    return fleet, codec, view


def _observation_space(fleet, view):
    config = fleet.config
    state = gym.spaces.Box(-np.inf, np.inf, (view.size if view else fleet.state_len,), np.float32)
    if config.obs == "state":
        return state
    w, h = config.image_size
    image = gym.spaces.Box(0, 255, (h, w), np.uint8)
    return image if config.obs == "image" else gym.spaces.Dict({"state": state, "image": image})


class HorizonVecEnv(VectorEnv):
    """`config.n` agents as one vector env: `step` sends every action, then waits for every
    observation (one game tick with frame skip 1: 20 steps per second).

    `HorizonVecEnv(FleetConfig(server=..., n=8, pairs=True))`, or the settings as keywords
    (`HorizonVecEnv(server=..., n=8)`). Action settings: `camera`, `buttons`, `hotbar`, `flat`,
    `delta_max`, `bins`, `max_turn`, `max_cps` (`ActionCodec`). `reward`: a function of the
    step's `infos` returning (n,) rewards.

    Episodes use Gymnasium's next-step autoreset, stretched: a reset takes game time (a respawn a
    few ticks, a restart seconds). After the step that ends agent `i`'s episode (`terminated`: it
    died; `truncated`: a fault, named in `infos["reset_reason"]`), its rows are no transitions
    until its new episode begins: its action is ignored, its observation is the last one, its
    reward 0, and `infos["active"][i]` is False. The step that returns the new episode's first
    observation has `infos["episode_start"][i]` True. `infos["valid"][i]` is True when row `i` is
    a real transition (the agent acted and this is what followed): train on those rows only.
    """

    metadata = {"render_modes": [], "autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(self, config=None, *, reward=None, fleet=None, **settings):
        self.fleet, codec, self.view = _open(config, fleet, None, settings)
        self.config = self.fleet.config
        self.num_envs = n = self.config.n
        self.reward_fn = reward
        self.state_fields = self.fleet.state_fields
        # Where each feature is in a state observation (the game's layout, or the chosen one).
        self.observation_fields = self.view.fields if self.view else self.state_fields
        self.items = self.fleet.items
        self.names = [slot.name for slot in self.fleet.slots]
        self.codec = ActionCodec(n, frame_skip=self.config.frame_skip, **codec)
        self.single_observation_space = _observation_space(self.fleet, self.view)
        self.single_action_space = self.codec.space
        self.observation_space = batch_space(self.single_observation_space, n)
        self.action_space = batch_space(self.single_action_space, n)
        self._states = np.zeros((n, self.fleet.state_len), np.float32)
        self._frames = None if self.fleet.frames is None else np.zeros_like(self.fleet.frames)
        self._ended = None  # episodes that ended in `release`, for the next step to report
        self.closed = False

    def _observations(self):
        obs = self.config.obs
        if obs == "image":
            return self._frames.copy()
        state = self.view.build(self._states) if self.view else self._states.copy()
        return state if obs == "state" else {"state": state, "image": self._frames.copy()}

    def _keep(self, states, rows):
        self._states[rows] = states[rows]
        if self._frames is not None:
            self._frames[rows] = self.fleet.frames[rows]

    def _infos(self, infos, active):
        out = {}
        for i, info in enumerate(infos):
            self._add_info(out, info, i)
        n = self.num_envs
        starts = out.get("episode_start", np.zeros(n, bool)).astype(bool)
        every = np.ones(n, bool)
        out.update(active=active.copy(), _active=every, episode_start=starts, _episode_start=every,
                   valid=active & ~starts, _valid=every)
        return out

    def reset(self, *, seed=None, options=None):
        """Starts a new episode for every agent and waits until all are alive in the world."""
        if seed is not None:
            self._np_random, self._np_random_seed = gym.utils.seeding.np_random(seed)
        states, infos = self.fleet.reset()
        self._ended = None
        every = np.ones(self.num_envs, bool)
        self._keep(states, every)
        out = self._infos(infos, every)
        out["valid"] = np.zeros(self.num_envs, bool)
        return self._observations(), out

    def release(self):
        """Lets go of every agent's keys and keeps its view: one step that is no transition.
        Call it before the trainer stops stepping for a while (a policy update): the game doesn't
        wait, and an agent left alone goes on with the keys of its last action. Released, the
        agents stand until the next `step`. An episode that ends meanwhile is reported by that
        step."""
        nan = float("nan")
        states, _, terminated, truncated, active = self.fleet.step([(0, 0, nan, nan)] * self.num_envs)
        self._keep(states, active)
        if self._ended is not None:
            terminated, truncated = terminated | self._ended[0], truncated | self._ended[1]
        self._ended = (terminated, truncated)

    def step(self, actions):
        buttons, hotbar, yaw, pitch = _encode(self.codec, actions, self._states, self.state_fields)
        states, infos, terminated, truncated, active = self.fleet.step(
            list(zip(buttons.tolist(), hotbar.tolist(), yaw.tolist(), pitch.tolist())))
        if self._ended is not None:
            terminated, truncated = terminated | self._ended[0], truncated | self._ended[1]
            self._ended = None
        self._keep(states, active)
        out = self._infos(infos, active)
        valid = out["valid"]
        rewards = np.zeros(self.num_envs, np.float32)
        # Rows without signals (no transition, or a fault cut the step) keep a reward of 0.
        paid = valid & out.get("_tick", False)
        if self.reward_fn is not None and paid.any():
            rewards[paid] = np.asarray(self.reward_fn(out), np.float32)[paid]
        return self._observations(), rewards, terminated, truncated, out

    def close_extras(self, **kwargs):
        self.fleet.close()


class HorizonEnv(gym.Env):
    """One agent. `HorizonEnv(server="HOST:PORT", names="Agent", target="Rival")`, or a
    `FleetConfig` (its `n` is taken as 1); the same action settings and `reward` as
    `HorizonVecEnv` (the reward function gets this agent's `info` dict).

    `step` ends an episode with `terminated` (the agent died) or `truncated` (a fault, named in
    `info["reset_reason"]`); `reset` then waits until the agent is alive in the world again: the
    respawn, the client joining again after a kick, or its process being restarted.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None, *, reward=None, fleet=None, reset_timeout=120.0, **settings):
        self.fleet, codec, self.view = _open(config, fleet, 1, settings)
        self.config = self.fleet.config
        if self.config.n != 1:
            self.fleet.close()
            raise ValueError("HorizonEnv is one agent: use HorizonVecEnv for more")
        self.reward_fn = reward
        self.reset_timeout = reset_timeout
        self.state_fields = self.fleet.state_fields
        self.observation_fields = self.view.fields if self.view else self.state_fields
        self.items = self.fleet.items
        self.name = self.fleet.slots[0].name
        self.codec = ActionCodec(1, frame_skip=self.config.frame_skip, **codec)
        self.observation_space = _observation_space(self.fleet, self.view)
        self.action_space = self.codec.space
        self._state = np.zeros(self.fleet.state_len, np.float32)
        self._frame = None if self.fleet.frames is None else np.zeros_like(self.fleet.frames[0])

    def _observation(self):
        obs = self.config.obs
        if obs == "image":
            return self._frame.copy()
        state = self.view.build(self._state[None])[0] if self.view else self._state.copy()
        return state if obs == "state" else {"state": state, "image": self._frame.copy()}

    def _keep(self, states):
        self._state[:] = states[0]
        if self._frame is not None:
            self._frame[:] = self.fleet.frames[0]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        states, infos = self.fleet.reset(timeout=self.reset_timeout)
        self._keep(states)
        return self._observation(), dict(infos[0], active=True)

    def step(self, action):
        buttons, hotbar, yaw, pitch = _encode(self.codec, action, self._state[None], self.state_fields)
        states, infos, terminated, truncated, active = self.fleet.step(
            [(int(buttons[0]), int(hotbar[0]), float(yaw[0]), float(pitch[0]))])
        info = dict(infos[0], active=bool(active[0]))
        reward = 0.0
        if active[0]:
            self._keep(states)
            if self.reward_fn is not None and not info.get("episode_start") and "tick" in info:
                reward = float(self.reward_fn(info))
        return self._observation(), reward, bool(terminated[0]), bool(truncated[0]), info

    def close(self):
        self.fleet.close()
