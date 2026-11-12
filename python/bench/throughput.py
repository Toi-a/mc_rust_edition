#!/usr/bin/env python3
"""Throughput of the Gymnasium vector env as N grows: what a trainer gets and what it costs.

    python/.venv/bin/python python/bench/throughput.py --ns 2,8,16,32 [--seconds 30]
    python/.venv/bin/python python/bench/throughput.py --ns 32 --obs image --per-process 8
    python/.venv/bin/python python/bench/throughput.py --ns 8,16 --server HOST:PORT

For each N it starts a `HorizonVecEnv` (agents in pairs) on a fresh local server (offline, a
normal world with a fixed seed; or on `--server`), plays `--policy` through the Gymnasium action
space (`chase`: the scripted fighter; `random`: samples of the action space), and after `--settle`
seconds measures for `--seconds`:

- steps/s per instance (20 = every game tick) and transitions/s in total (valid rows only: an
  agent between episodes gives none);
- per instance: CPU % of one core (all threads) and RSS / PSS in MB; with `--per-process K`, a
  process's numbers divided by K;
- the Python side: CPU % of one core and milliseconds of it per vector step (the env and the
  scripted policy, no network wait);
- late ticks per 1000 agent steps (a tick that ran before its action arrived).

A row "holds 20 Hz" with at least 19.9 steps/s and under 5 late ticks per 1000 steps. Prints a
table, and a JSON line per N to `--out`.
"""
import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
from fleet import cpu_seconds, gpu_memory_mb, mean, memory_mb, rss_mb  # noqa: E402  (bench/fleet.py)

from horizon_rl import HorizonVecEnv, rewards  # noqa: E402
from horizon_rl.actions import from_bits  # noqa: E402
from horizon_rl.fleet import FleetConfig  # noqa: E402
from horizon_rl.local import LocalServer, remove  # noqa: E402
from horizon_rl.scripted import Chaser  # noqa: E402


def run(n, args):
    root = Path(tempfile.mkdtemp(prefix=f"hz-thr-{n}-", dir="/tmp"))
    server = None
    if not args.server:
        server = LocalServer(root, properties={"max-players": str(max(64, n + 4)), "level-seed": "1234",
                                               "level-type": "minecraft:normal"})
        server.command("gamerule spawn_radius 0")
    image = tuple(int(v) for v in args.image.lower().split("x"))
    config = FleetConfig(server=args.server or server.address, n=n, pairs=True, obs=args.obs,
                         image_size=image, frame_skip=args.frame_skip, root=str(root / "fleet"),
                         clients_per_process=args.per_process, start_timeout=120 + n)
    gpu0 = gpu_memory_mb()
    env = None
    try:
        t_start = time.monotonic()
        env = HorizonVecEnv(config, reward=rewards.duel, camera="absolute" if args.policy == "chase" else "delta")
        started_in = time.monotonic() - t_start
        if server:
            for name in env.names:
                server.command(f"give {name} minecraft:diamond_sword")
        obs, _ = env.reset()
        env.action_space.seed(0)
        chasers = [Chaser(env.state_fields, seed=i, sword_slot=1) for i in range(n)]
        state_of = (lambda o: o["state"]) if args.obs == "both" else (lambda o: o)
        # With images only, the scripted fighter has no state to read: it gets the env's copy.
        if args.obs == "image":
            state_of = lambda o: env._states  # noqa: E731

        def act(obs):
            if args.policy == "random":
                return env.action_space.sample()
            rows = [from_bits(*p.act(s)) for p, s in zip(chasers, state_of(obs))]
            return {k: np.stack([r[k] for r in rows]) for k in rows[0]}

        stats = dict(steps=0, transitions=0, late=0.0, agent_steps=0, episodes=0, reward=0.0)

        def play(duration, stats=None):
            nonlocal obs
            end = time.monotonic() + duration
            while time.monotonic() < end:
                obs, reward, term, trunc, infos = env.step(act(obs))
                if stats is None:
                    continue
                valid = infos["valid"]
                stats["steps"] += 1
                stats["transitions"] += int(valid.sum())
                stats["episodes"] += int((term | trunc).sum())
                stats["reward"] += float(np.abs(reward).sum())
                if "late_ticks" in infos:
                    has = infos["_late_ticks"]
                    stats["late"] += float(infos["late_ticks"][has].sum())
                    stats["agent_steps"] += int(has.sum())

        play(args.settle)
        pids = list(dict.fromkeys(s.proc.proc.pid for s in env.fleet.slots))
        k = n / len(pids)
        cpu0 = [cpu_seconds(p) for p in pids]
        py0, t0 = time.process_time(), time.monotonic()
        rss = []
        while time.monotonic() - t0 < args.seconds:
            play(min(2.0, args.seconds - (time.monotonic() - t0)), stats)
            rss += [r / k for r in map(rss_mb, pids) if r is not None]
        dt, py = time.monotonic() - t0, time.process_time() - py0
        cpu = [(c1 - c0) / dt * 100 / k for c0, c1 in zip(cpu0, (cpu_seconds(p) for p in pids))
               if c0 is not None and c1 is not None]
        pss = [m[1] / k for m in map(memory_mb, pids) if m]
        gpu1 = gpu_memory_mb()
        steps_per_s = stats["steps"] / dt
        late = 1000 * stats["late"] / max(stats["agent_steps"], 1)
        return {
            "n": n, "obs": args.obs, "per_process": args.per_process, "policy": args.policy,
            "frame_skip": args.frame_skip, "started_s": round(started_in, 1),
            "steps_per_s_per_instance": round(steps_per_s, 2),
            "transitions_per_s_total": round(stats["transitions"] / dt, 1),
            "client_cpu_pct": round(mean(cpu), 2),
            "clients_cpu_pct_total": round(sum(cpu) * k, 1),
            "client_rss_mb": round(mean(rss), 1),
            "client_pss_mb": round(mean(pss), 1),
            "gpu_mb_per_client": (round((gpu1 - gpu0) / n, 1)
                                  if args.obs != "state" and gpu0 is not None and gpu1 is not None else None),
            "python_cpu_pct": round(py / dt * 100, 1),
            "python_ms_per_step": round(1000 * py / max(stats["steps"], 1), 2),
            "late_ticks_per_1000": round(late, 1),
            "episodes_ended": stats["episodes"],
            "restarts": sum(s.restarts for s in env.fleet.slots),
            "holds_20hz": bool(steps_per_s * args.frame_skip >= 19.9 and late < 5),
        }
    finally:
        if env is not None:
            env.close()
        if server is not None:
            server.stop()
        remove(root)


COLUMNS = [("n", "N"), ("steps_per_s_per_instance", "steps/s each"), ("transitions_per_s_total", "transitions/s"),
           ("client_cpu_pct", "CPU % each"), ("client_rss_mb", "RSS MB"), ("client_pss_mb", "PSS MB"),
           ("gpu_mb_per_client", "GPU MB"), ("python_cpu_pct", "Python CPU %"),
           ("python_ms_per_step", "Python ms/step"), ("late_ticks_per_1000", "late/1000"),
           ("holds_20hz", "20 Hz")]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="2,8,16")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--settle", type=float, default=10)
    ap.add_argument("--server", help="HOST:PORT of a running server (default: a local stand-in per N)")
    ap.add_argument("--obs", default="state", choices=["state", "image", "both"])
    ap.add_argument("--image", default="64x64")
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--per-process", type=int, default=1)
    ap.add_argument("--policy", default="chase", choices=["chase", "random"])
    ap.add_argument("--out")
    args = ap.parse_args()
    results = []
    for n in (int(x) for x in args.ns.split(",")):
        result = run(n, args)
        results.append(result)
        print(json.dumps(result), flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(json.dumps(result) + "\n")
    print("\n| " + " | ".join(title for _, title in COLUMNS) + " |")
    print("|" + "---|" * len(COLUMNS))
    for r in results:
        cells = ["–" if r[key] is None else ("yes" if r[key] is True else "no" if r[key] is False else str(r[key]))
                 for key, _ in COLUMNS]
        print("| " + " | ".join(cells) + " |")
    held = [r["n"] for r in results if r["holds_20hz"]]
    print(f"\nlargest N that held 20 Hz: {max(held) if held else 'none'}")


if __name__ == "__main__":
    main()
