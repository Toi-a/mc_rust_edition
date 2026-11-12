#!/usr/bin/env python3
"""M4 end to end: the Gymnasium envs on real clients and a local server. Needs the fast build,
.local/ and a GPU (it asks for images too).

    python/.venv/bin/python python/tests/e2e_gym.py

Checks, on a `HorizonVecEnv` of 4 agents in pairs with state and image observations: resets and
steps give observations inside the declared spaces at 20 steps per second; random actions from
the action space are played; scripted fights through the Gymnasium actions land hits, and the
duel reward is the damage dealt minus the damage taken; a death is `terminated`, then the agent is no
transition until its new episode starts, while the others' rows stay valid; a kick is
`truncated` with its reason; a staged action space (four keys, aim assist) keeps each agent
facing its opponent. Then a one-agent `HorizonEnv` with the delta camera: turns add up,
a death terminates and `reset` brings the agent back alive.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from horizon_rl.actions import from_bits  # noqa: E402
from horizon_rl.env import HorizonEnv, HorizonVecEnv  # noqa: E402
from horizon_rl.fleet import FleetConfig  # noqa: E402
from horizon_rl.local import LocalServer, remove  # noqa: E402
from horizon_rl.rewards import duel  # noqa: E402
from horizon_rl.scripted import Chaser  # noqa: E402

failures = []
N = 4


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def batch(actions):
    return {k: np.stack([a[k] for a in actions]) for k in actions[0]}


def vector(server, root):
    config = FleetConfig(server=server.address, n=N, pairs=True, obs="both", image_size=(64, 48),
                         root=str(root / "fleet"))
    env = HorizonVecEnv(config, reward=duel)
    try:
        for name in env.names:
            server.command(f"give {name} minecraft:diamond_sword")
        obs, infos = env.reset()
        check(env.observation_space.contains(obs), "the reset observations are in the observation space")
        check(obs["image"].shape == (N, 48, 64) and obs["state"].shape[0] == N,
              f"batched: state {obs['state'].shape}, image {obs['image'].shape}")
        check(not infos["valid"].any() and infos["episode_start"].all(), "a reset starts every episode")

        # Random actions, as a first policy would send them.
        env.action_space.seed(1)
        t0, steps, inside, moved = time.monotonic(), 0, True, obs["state"].copy()
        while time.monotonic() - t0 < 4:
            obs, rewards, term, trunc, infos = env.step(env.action_space.sample())
            inside &= env.observation_space.contains(obs)
            steps += 1
        rate = steps / (time.monotonic() - t0)
        check(inside, "every step's observations are in the space")
        check(19 <= rate <= 21, f"one step per tick ({rate:.1f} steps/s)")
        pos = env.state_fields["self_pos"]
        check((np.abs(obs["state"][:, pos] - moved[:, pos]).sum(axis=1) > 0.5).all(),
              "random actions moved every agent")
        check(obs["image"].std(axis=(1, 2)).min() > 5, "every frame is a picture")

        # Scripted fights, through the Gymnasium action space.
        policies = [Chaser(env.state_fields, seed=i, sword_slot=1) for i in range(N)]
        state = {"obs": obs, "damage": 0.0, "reward_off": 0, "hits": 0, "ended": [[] for _ in range(N)],
                 "started": np.zeros(N, int)}

        def play(seconds, until=None, watch=None):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                actions = batch([from_bits(*p.act(s)) for p, s in zip(policies, state["obs"]["state"])])
                obs, rewards, term, trunc, infos = env.step(actions)
                state["obs"] = obs
                if "target_damage" in infos:
                    # (Two fighters often hit in the same tick: dealt minus taken is then 0.)
                    valid = infos["valid"]
                    dealt, taken = np.nan_to_num(infos["target_damage"]), infos["damage_taken"]
                    state["damage"] += float((dealt + taken)[valid].sum())
                    state["reward_off"] += int(np.abs(rewards - np.where(valid, dealt - taken, 0)).max() > 1e-4)
                state["hits"] += int(infos["hit_landed"][infos["valid"]].sum()) if "hit_landed" in infos else 0
                state["started"] += infos["episode_start"]
                for i in range(N):
                    if term[i]:
                        state["ended"][i].append("died")
                    if trunc[i]:
                        state["ended"][i].append(str(infos["reset_reason"][i]))
                if watch is not None:
                    watch(infos, rewards)
                if until is not None and until():
                    return True
            return False

        play(8, until=lambda: state["hits"] >= 4)
        check(state["hits"] >= 4, f"scripted pairs fight through the env ({state['hits']} hits)")
        check(state["damage"] > 0 and state["reward_off"] == 0,
              f"each step's reward is the damage dealt minus the damage taken ({state['damage']:.1f} points moved)")

        # A death: terminated, then no transition until the new episode; the others play on.
        play(20, until=lambda: all(env.fleet.slots[i].state == "ready" and not env.fleet.slots[i].episode_start
                                   for i in range(N)))
        seen = {"away": 0, "start_reward": None, "others_valid": True}
        victim = 3
        state["ended"] = [[] for _ in range(N)]
        before = state["started"][victim]

        def watch(infos, rewards):
            if state["ended"][victim]:
                seen["away"] += int(not infos["active"][victim])
                if infos["episode_start"][victim]:
                    seen["start_reward"] = float(rewards[victim])
                    seen["start_valid"] = bool(infos["valid"][victim])
        server.command(f"kill {env.names[victim]}")
        back = play(20, until=lambda: state["started"][victim] > before, watch=watch)
        check("died" in state["ended"][victim], f"a death terminates ({state['ended'][victim]})")
        check(back, "its next episode starts by itself")
        check(seen["away"] >= 1, f"between the two it is inactive ({seen['away']} steps)")
        check(seen.get("start_valid") is False and seen["start_reward"] == 0.0,
              "the new episode's first observation is no transition and carries no reward")
        check(env.observation_space.contains(state["obs"]), "observations stay in the space through it")

        # A kick: truncated, with its reason; the client joins again by itself.
        state["ended"] = [[] for _ in range(N)]
        before = state["started"][0]
        server.command(f"kick {env.names[0]} testing")
        back = play(60, until=lambda: state["started"][0] > before)
        check(back and any(r.startswith("disconnected") for r in state["ended"][0]),
              f"a kick truncates with its reason and the agent comes back ({state['ended'][0]})")

        # The first training stage's actions on the same clients: four keys, aim assist, flat.
        staged = HorizonVecEnv(fleet=env.fleet, camera="aim", buttons=("forward", "back", "jump", "attack"),
                               hotbar=False, flat=True)
        check(staged.single_action_space.shape == (4,), f"a staged action space ({staged.single_action_space})")
        obs, _ = staged.reset()
        for _ in range(40):  # opponents show up a few ticks after the respawn
            obs, _, _, _, infos = staged.step(np.zeros((N, 4), int))
        present = obs["state"][:, env.state_fields["target_present"].start] > 0.5
        off = np.abs(obs["state"][:, env.state_fields["target_bearing"]][present])
        check(present.sum() >= 2 and off.max() < 1.0,
              f"aim assist keeps every agent facing its opponent ({present.sum()} agents, off by {off.max():.2f} degrees)")
    finally:
        env.close()


def single(server, root):
    env = HorizonEnv(server=server.address, names="Solo", camera="delta", delta_max=20.0,
                     root=str(root / "solo"))
    try:
        obs, info = env.reset()
        check(env.observation_space.contains(obs) and info["alive"] == 1, "one agent: alive after reset")
        yaw0 = float(obs[env.state_fields["self_yaw"]][0])
        action = env.action_space.sample()
        action["buttons"][:] = 0
        action["hotbar"] = 0
        action["look"][:] = [15.0, 4.0]
        for _ in range(3):
            obs, reward, term, trunc, info = env.step(action)
        yaw = float(obs[env.state_fields["self_yaw"]][0])
        pitch = float(obs[env.state_fields["self_pitch"]][0])
        turned = (yaw - yaw0 + 180) % 360 - 180
        check(abs(turned - 45.0) < 0.01 and abs(pitch - 12.0) < 0.01,
              f"three delta turns add up (yaw {turned:+.2f}, pitch {pitch:.2f})")
        server.command("kill Solo")
        term = False
        for _ in range(100):
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc:
                break
        check(term, "its death terminates the episode")
        obs, info = env.reset()
        check(info["alive"] == 1 and env.observation_space.contains(obs), "reset waits for the respawn")
    finally:
        env.close()


def main():
    # A short path: the clients' sockets live under it (SUN_LEN).
    root = Path(tempfile.mkdtemp(prefix="hz-gym-", dir="/tmp"))
    server = LocalServer(root)
    try:
        vector(server, root)
        single(server, root)
    finally:
        server.stop()
    if failures:
        print(f"\n{len(failures)} failed; logs kept in {root}")
        sys.exit(1)
    print("\nall passed")
    remove(root)


if __name__ == "__main__":
    main()
