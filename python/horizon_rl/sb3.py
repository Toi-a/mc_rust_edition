"""A Stable-Baselines3 `VecEnv` over a `HorizonVecEnv` (`pip install stable-baselines3`).

    env = HorizonVecEnv(server=..., n=16, pairs=True, flat=True, camera="aim", hotbar=False,
                        buttons=("forward", "back", "jump", "attack"), reward=rewards.duel)
    model = PPO("MlpPolicy", VecMonitor(HorizonSB3VecEnv(env))).learn(1_000_000)

SB3 takes no `Dict` action space: make the env with `flat=True`. Every agent is one of SB3's
envs, all trained as one policy, so agents paired on the server play against themselves.

What SB3's API changes:
- Images get a channel axis, (1, H, W), which is what its CNN policies read.
- `infos` is a list of dicts: each agent's signals, `valid`, `active`, `episode_start`, and at
  an episode's end `terminal_observation` and `TimeLimit.truncated` (a fault, not a death).
- SB3 expects an env that ended to be in its next episode at once. Here the respawn takes about
  a second of game time, and the other agents can't wait for it. Until the agent is back its
  rows repeat its last observation with reward 0 and `info["valid"]` False, and SB3 stores them
  as the first steps of the next episode: about 20 steps per death in which nothing the policy
  does has an effect. They dilute a rollout a little and shift no reward.
- The game runs on while SB3 updates the policy between rollouts (seconds without a step).
  Pass `callback=ReleaseKeys()` to `learn`: it lets go of every agent's keys at the end of each
  rollout, so they stand still through the update instead of running on with their last action.
  The first step after it reports the pause in `info["late_ticks"]`.
"""
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


def _with_channel(space):
    if isinstance(space, spaces.Dict):
        return spaces.Dict({k: _with_channel(s) for k, s in space.spaces.items()})
    if space.dtype == np.uint8 and len(space.shape) == 2:
        return spaces.Box(0, 255, (1, *space.shape), np.uint8)
    return space


class HorizonSB3VecEnv(VecEnv):
    def __init__(self, env):
        if isinstance(env.single_action_space, spaces.Dict):
            raise ValueError("Stable-Baselines3 takes no Dict action space: make the env with flat=True")
        self.env = env
        self._image = env.config.obs == "image"
        self._both = env.config.obs == "both"
        self._actions = None
        super().__init__(env.num_envs, _with_channel(env.single_observation_space), env.single_action_space)

    def _obs(self, obs):
        if self._image:
            return obs[:, None]
        if self._both:
            return {"state": obs["state"], "image": obs["image"][:, None]}
        return obs

    def _row(self, obs, i):
        return {k: v[i] for k, v in obs.items()} if isinstance(obs, dict) else obs[i]

    def _info_list(self, infos):
        keys = [k for k in infos if not k.startswith("_")]
        return [{k: infos[k][i] for k in keys if infos[f"_{k}"][i]} for i in range(self.num_envs)]

    def reset(self):
        obs, infos = self.env.reset()
        self.reset_infos = self._info_list(infos)
        return self._obs(obs)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        obs, rewards, terminated, truncated, infos = self.env.step(self._actions)
        obs = self._obs(obs)
        dones = terminated | truncated
        out = self._info_list(infos)
        for i in np.flatnonzero(dones):
            out[i]["terminal_observation"] = self._row(obs, i)
            out[i]["TimeLimit.truncated"] = bool(truncated[i] and not terminated[i])
        return obs, rewards, dones, out

    def close(self):
        self.env.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self.env, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self.env, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        # One env behind every index: called once.
        result = getattr(self.env, method_name)(*args, **kwargs)
        return [result for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]


class ReleaseKeys(BaseCallback):
    """Lets go of the agents' keys while the policy is updated (`HorizonVecEnv.release`)."""

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        self.training_env.env_method("release", indices=[0])
