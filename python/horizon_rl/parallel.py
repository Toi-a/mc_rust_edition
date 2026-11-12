"""An optional PettingZoo `ParallelEnv` view of a `HorizonVecEnv` (`pip install pettingzoo`),
for multi-agent trainers that want agents by name.

The agents are the fleet's players. The server pairs them up, so the env only carries each one's
own observation, action and signals. Unlike a turn-based PettingZoo env, the world doesn't stop:
an agent whose episode ended leaves `agents` and comes back by itself when its next episode
starts (its first observation is in that step's results, with `infos[name]["episode_start"]`).
`step` returns entries only for agents that acted or just started; `reset` restarts everyone.
"""
import numpy as np

from .env import HorizonVecEnv

try:
    from pettingzoo import ParallelEnv
except ImportError:  # usable without PettingZoo: the same methods, no base class
    ParallelEnv = object


class HorizonParallelEnv(ParallelEnv):
    metadata = {"name": "horizon_duel_v0", "render_modes": [], "is_parallelizable": True}

    def __init__(self, config=None, *, vec_env=None, **settings):
        self.vec = vec_env if vec_env is not None else HorizonVecEnv(config, **settings)
        self.possible_agents = list(self.vec.names)
        self.agents = []
        self._noop = self.vec.codec.placeholder()  # for agents between episodes: it is ignored

    def observation_space(self, agent):
        return self.vec.single_observation_space

    def action_space(self, agent):
        return self.vec.single_action_space

    def _row(self, obs, i):
        return {k: v[i] for k, v in obs.items()} if isinstance(obs, dict) else obs[i]

    def _info(self, infos, i):
        # Every key has its `_key` mask: which agents have a value this step.
        return {k: v[i] for k, v in infos.items() if not k.startswith("_") and infos[f"_{k}"][i]}

    def reset(self, seed=None, options=None):
        obs, infos = self.vec.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)
        return ({a: self._row(obs, i) for i, a in enumerate(self.agents)},
                {a: self._info(infos, i) for i, a in enumerate(self.agents)})

    def step(self, actions):
        """`actions`: name -> action, for the agents in `agents` (others' are ignored)."""
        rows = [actions.get(a, self._noop) for a in self.possible_agents]
        if self.vec.codec.flat:
            batch = np.stack([np.asarray(r) for r in rows])
        else:
            batch = {k: np.stack([np.asarray(r[k]) for r in rows]) for k in self._noop}
        obs, rewards, terminated, truncated, infos = self.vec.step(batch)
        out = ({}, {}, {}, {}, {})
        agents = []
        for i, a in enumerate(self.possible_agents):
            if not infos["active"][i]:
                continue
            out[0][a] = self._row(obs, i)
            out[1][a] = float(rewards[i])
            out[2][a] = bool(terminated[i])
            out[3][a] = bool(truncated[i])
            out[4][a] = self._info(infos, i)
            if not (terminated[i] or truncated[i]):
                agents.append(a)
        self.agents = agents
        return out

    def close(self):
        self.vec.close()
