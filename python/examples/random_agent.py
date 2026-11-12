#!/usr/bin/env python3
"""A random agent on the Gymnasium vector env: the smallest training loop's shape.

    python/.venv/bin/python python/examples/random_agent.py --server HOST:PORT -n 8
    python/.venv/bin/python python/examples/random_agent.py --local -n 4 --obs both --camera bins

`--local` starts the repo's own offline server as a stand-in. Agents 2k and 2k+1 are each
other's opponent. Every second it prints the step rate, the transitions collected, ended
episodes and the reward (damage dealt minus damage taken).
"""
import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl import HorizonVecEnv, rewards  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", help="HOST:PORT of an offline-mode 26.3 server")
    ap.add_argument("--local", action="store_true", help="start a local stand-in server instead")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--obs", default="state", choices=["state", "image", "both"])
    ap.add_argument("--camera", default="delta", choices=["absolute", "delta", "bins"])
    ap.add_argument("--per-process", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=30)
    args = ap.parse_args()
    if not args.server and not args.local:
        ap.error("--server HOST:PORT or --local")

    server = None
    if args.local:
        from horizon_rl.local import LocalServer
        server = LocalServer(tempfile.mkdtemp(prefix="hz-random-", dir="/tmp"))
    env = None
    try:
        env = HorizonVecEnv(server=server.address if server else args.server, n=args.n, pairs=True,
                            obs=args.obs, clients_per_process=args.per_process,
                            camera=args.camera, reward=rewards.duel)
        print("observation space:", env.single_observation_space)
        print("action space:", env.single_action_space)
        obs, infos = env.reset(seed=0)
        env.action_space.seed(0)
        started = last = time.monotonic()
        steps = transitions = episodes = 0
        returns = 0.0
        while time.monotonic() - started < args.seconds:
            actions = env.action_space.sample()          # a policy would map obs -> actions here
            obs, reward, terminated, truncated, infos = env.step(actions)
            valid = infos["valid"]                       # rows that are real transitions
            steps += 1
            transitions += int(valid.sum())
            episodes += int((terminated | truncated).sum())
            returns += float(reward[valid].sum())
            now = time.monotonic()
            if now - last >= 1:
                print(f"{steps / (now - last):5.1f} steps/s  {transitions} transitions  "
                      f"{episodes} episodes ended  reward {returns:+.1f}", flush=True)
                last, steps, transitions, episodes, returns = now, 0, 0, 0, 0.0
    finally:
        if env is not None:
            env.close()
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
