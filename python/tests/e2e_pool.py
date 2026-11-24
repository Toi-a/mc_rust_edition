#!/usr/bin/env python3
"""M6 end to end: PPO against the opponent pool on a server with the round plugin
(docs/rl/design.md#rounds-and-the-opponent-pool-m6-planned). Needs the fast build, .local/, and
stable-baselines3 (~/.cache/horizon-rl-sb3-venv).

    python python/tests/e2e_pool.py --server 127.0.0.1:25565 --obs state
    python python/tests/e2e_pool.py --server 127.0.0.1:25565 --obs image

The pool lists two hand-made versions, real SB3 models whose action head is set by hand: `brawler`
walks at its opponent (aim assist) and swings about once per sword cooldown, `statue` holds
nothing. The learner starts as a brawler, so rounds end in kills rather than at the time cap. It
trains for `--steps` learner steps with snapshots every `--snapshot-every`, and checks:
rounds end with results; each pair's opponent is drawn again at round ends and differs between
rounds; the stats (CSV) add up to the rounds seen; the learner beats the statue nearly always;
the run's snapshots join the pool and are drawn; every agent stays in play at 20 Hz.
"""
import argparse
import collections
import csv
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402

from horizon_rl import HorizonVecEnv, rewards  # noqa: E402
from horizon_rl.observations import PRESETS  # noqa: E402
from horizon_rl.pool import Pool, PoolConfig  # noqa: E402
from horizon_rl.pool_sb3 import OpponentPoolVecEnv, PoolSnapshots, PoolStats  # noqa: E402
from horizon_rl.sb3 import HorizonSB3VecEnv, ReleaseKeys  # noqa: E402

failures = []
BUTTONS = ("forward", "back", "jump", "attack")
POLICY = {"state": "MlpPolicy", "image": "CnnPolicy", "both": "MultiInputPolicy"}


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def logit(p):
    return math.log(p / (1 - p))


def set_head(model, presses):
    """The action head ignores the observation and presses each button with its probability."""
    net = model.policy.action_net
    with torch.no_grad():
        net.weight.zero_()
        bias = []
        for name in BUTTONS:
            bias += [0.0, logit(presses.get(name, 0.002))]
        net.bias.copy_(torch.tensor(bias))
    return model


class Rounds(BaseCallback):
    """Each pair's opponents, round by round, and the results the learner saw."""

    def __init__(self, env):
        super().__init__()
        self.env = env
        self.by_pair = collections.defaultdict(list)  # pair -> [(version name, result)]
        self.inactive = 0
        self.steps = 0
        self.started = None

    def _on_step(self):
        if self.started is None:
            self.started = time.monotonic()
        self.steps += 1
        for pair, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            if not info.get("active", True):
                self.inactive += 1
            if done:
                result = next((r for r in ("win", "loss", "draw", "timeout", "abort")
                               if info.get("round_" + r, 0) > 0), None)
                self.by_pair[pair].append((info.get("opponent_version"), result))
        return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True, help="HOST:PORT of a server with the round plugin")
    ap.add_argument("--obs", default="state", choices=sorted(POLICY))
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000, help="learner steps (all seats)")
    ap.add_argument("--snapshot-every", type=int, default=1500)
    args = ap.parse_args()

    root = Path(tempfile.mkdtemp(prefix="hz-pool-e2e-", dir="/tmp"))
    run = root / "run"
    observation = tuple(PRESETS["sword"]) + ("round_time_left",)
    env = HorizonVecEnv(server=args.server, n=args.n, pairs=True, rounds=True, obs=args.obs, flat=True,
                        camera="aim", hotbar=False, buttons=BUTTONS, observation=observation,
                        target_health=True, reward=rewards.Round(), root=str(root / "fleet"))
    try:
        inner = HorizonSB3VecEnv(env)
        policy = POLICY[args.obs]
        brawler = {"forward": 0.98, "attack": 0.08}
        versions = {"brawler": brawler, "statue": {}}
        for name, presses in versions.items():
            set_head(PPO(policy, inner, device="cpu"), presses).save(root / f"{name}.zip")
        config = PoolConfig(snapshot_every=args.snapshot_every, max_size=4,
                            versions=tuple((str(root / f"{name}.zip"), 1.0) for name in versions))
        pool_env = OpponentPoolVecEnv(inner, Pool(config), opponent_device="cpu")
        check(len(pool_env.models) == 2 and pool_env.num_envs == args.n // 2,
              f"two listed versions loaded, {pool_env.num_envs} learner seats")
        model = set_head(PPO(policy, VecMonitor(pool_env), n_steps=256, batch_size=256, n_epochs=1,
                             learning_rate=1e-5, device="cpu"), brawler)
        rounds = Rounds(pool_env)
        t0 = time.monotonic()
        model.learn(args.steps, callback=[ReleaseKeys(), PoolSnapshots(run, verbose=1), PoolStats(run), rounds])
        took = time.monotonic() - t0

        results = [r for pair in rounds.by_pair.values() for _, r in pair]
        print("rounds per pair:", {p: v for p, v in rounds.by_pair.items()})
        check(len(results) >= 2 * pool_env.num_envs, f"rounds ended ({len(results)})")
        check(all(r is not None for r in results), f"each with a result ({collections.Counter(results)})")
        swapped = [len({v for v, _ in pair}) > 1 for pair in rounds.by_pair.values()]
        check(len(swapped) == pool_env.num_envs and all(swapped), "every pair met more than one version")

        with open(run / "pool_stats.csv") as f:
            rows = {r["version"]: r for r in csv.DictReader(f)}
        counted = sum(int(r["rounds"]) for r in rows.values())
        check(counted == len(results) == pool_env.rounds_done,
              f"the stats count every round ({counted} in the CSV, {len(results)} seen)")
        per_version = collections.Counter(v for pair in rounds.by_pair.values() for v, _ in pair)
        check(all(int(rows[v]["rounds"]) == c for v, c in per_version.items()),
              f"per version too ({dict(per_version)})")
        statue = rows.get("statue")
        if statue:
            wins, n = int(statue["wins"]), int(statue["rounds"])
            check(wins >= 0.8 * n, f"the learner beats the statue ({wins} of {n})")
        else:
            check(False, "the statue was drawn")
        snapshots = sorted(p.name for p in (run / "snapshots").glob("*.zip"))
        check(len(snapshots) >= 2, f"snapshots were taken ({snapshots})")
        check(any(v.startswith("s0") for v in per_version), f"a snapshot was drawn ({sorted(per_version)})")
        check(pool_env.rounds_done and len(pool_env.pool) <= config.max_size,
              f"the pool stays within max_size ({len(pool_env.pool)})")
        rate = rounds.steps / took
        check(rate > 17, f"about 20 steps per second with the policy updates ({rate:.1f})")
        for row in rows.values():
            print("  ", {k: row[k] for k in ("version", "rounds", "wins", "losses", "draws", "timeouts", "aborts")})
    finally:
        env.close()
    if failures:
        print(f"{len(failures)} FAILED: {failures}")
        sys.exit(1)
    print("all ok; files in", root)


if __name__ == "__main__":
    main()
