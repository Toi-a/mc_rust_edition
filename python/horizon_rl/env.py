"""Gymnasium environments over a `Fleet` (docs/rl/README.md#gymnasium).

- `HorizonVecEnv`: N agents stepped together, each an independent player on the server.
- `HorizonEnv`: one agent.

Observations follow `FleetConfig.obs`: `"state"` a float32 vector (the fair-play view; the layout
is `env.state_fields`, name -> slice), `"image"` an (H, W) uint8 frame, `"both"` a dict of the
two. Actions: `horizon_rl.actions.ActionCodec` (buttons, hotbar, camera). Rewards: 0 unless a
`reward` function is given (`horizon_rl.rewards`); every signal is in `info`.

The server runs in real time, so nothing here blocks on one agent: an agent whose episode ended
(a death, or a fault: kick, crash, hang) is reset by its supervisor while the others play on.
"""
import dataclasses

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space

from .actions import DEFAULT_BINS, ActionCodec
from .fleet import Fleet, FleetConfig

_CODEC_DEFAULTS = {"camera": "absolute", "delta_max": 30.0, "bins": DEFAULT_BINS, "max_turn": 0.0,
                   "max_cps": 0.0}


def _open(config, fleet, n, settings):
    """The started fleet of an env, and the action settings left in `settings`."""
    codec = {k: settings.pop(k, v) for k, v in _CODEC_DEFAULTS.items()}
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
    except BaseException:
        fleet.close()
        raise
    return fleet, codec


def _observation_space(fleet):
    config = fleet.config
    state = gym.spaces.Box(-np.inf, np.inf, (fleet.state_len,), np.float32)
    if config.obs == "state":
        return state
    w, h = config.image_size
    image = gym.spaces.Box(0, 255, (h, w), np.uint8)
    return image if config.obs == "image" else gym.spaces.Dict({"state": state, "image": image})


class HorizonVecEnv(VectorEnv):
    """`config.n` agents as one vector env: `step` sends every action, then waits for every
    observation (one game tick with frame skip 1: 20 steps per second).

    `HorizonVecEnv(FleetConfig(server=..., n=8, pairs=True))`, or the settings as keywords
    (`HorizonVecEnv(server=..., n=8)`). Action settings: `camera`, `delta_max`, `bins`,
    `max_turn`, `max_cps` (`ActionCodec`). `reward`: a function of the step's `infos`
    returning (n,) rewards.

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
        self.fleet, codec = _open(config, fleet, None, settings)
        self.config = self.fleet.config
        self.num_envs = n = self.config.n
        self.reward_fn = reward
        self.state_fields = self.fleet.state_fields
        self.items = self.fleet.items
        self.names = [slot.name for slot in self.fleet.slots]
        self.codec = ActionCodec(n, frame_skip=self.config.frame_skip, **codec)
        self.single_observation_space = _observation_space(self.fleet)
        self.single_action_space = self.codec.space
        self.observation_space = batch_space(self.single_observation_space, n)
        self.action_space = batch_space(self.single_action_space, n)
        self._yaw = self.state_fields["self_yaw"].start
        self._pitch = self.state_fields["self_pitch"].start
        self._states = np.zeros((n, self.fleet.state_len), np.float32)
        self._frames = None if self.fleet.frames is None else np.zeros_like(self.fleet.frames)
        self.closed = False

    def _observations(self):
        obs = self.config.obs
        if obs == "state":
            return self._states.copy()
        if obs == "image":
            return self._frames.copy()
        return {"state": self._states.copy(), "image": self._frames.copy()}

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
        every = np.ones(self.num_envs, bool)
        self._keep(states, every)
        out = self._infos(infos, every)
        out["valid"] = np.zeros(self.num_envs, bool)
        return self._observations(), out

    def step(self, actions):
        buttons, hotbar, yaw, pitch = self.codec.encode(
            actions, self._states[:, self._yaw], self._states[:, self._pitch])
        states, infos, terminated, truncated, active = self.fleet.step(
            list(zip(buttons.tolist(), hotbar.tolist(), yaw.tolist(), pitch.tolist())))
        self._keep(states, active)
        out = self._infos(infos, active)
        valid = out["valid"]
        rewards = np.zeros(self.num_envs, np.float32)
        if self.reward_fn is not None and valid.any():
            # Rows that are no transition have no signals: their reward stays 0.
            rewards[valid] = np.asarray(self.reward_fn(out), np.float32)[valid]
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
        self.fleet, codec = _open(config, fleet, 1, settings)
        self.config = self.fleet.config
        if self.config.n != 1:
            self.fleet.close()
            raise ValueError("HorizonEnv is one agent: use HorizonVecEnv for more")
        self.reward_fn = reward
        self.reset_timeout = reset_timeout
        self.state_fields = self.fleet.state_fields
        self.items = self.fleet.items
        self.name = self.fleet.slots[0].name
        self.codec = ActionCodec(1, frame_skip=self.config.frame_skip, **codec)
        self.observation_space = _observation_space(self.fleet)
        self.action_space = self.codec.space
        self._state = np.zeros(self.fleet.state_len, np.float32)
        self._frame = None if self.fleet.frames is None else np.zeros_like(self.fleet.frames[0])

    def _observation(self):
        obs = self.config.obs
        if obs == "state":
            return self._state.copy()
        if obs == "image":
            return self._frame.copy()
        return {"state": self._state.copy(), "image": self._frame.copy()}

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
        yaw = self._state[self.state_fields["self_yaw"]]
        pitch = self._state[self.state_fields["self_pitch"]]
        buttons, hotbar, yaw, pitch = self.codec.encode(action, yaw, pitch)
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
