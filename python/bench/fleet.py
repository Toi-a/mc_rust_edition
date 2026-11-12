#!/usr/bin/env python3
"""CPU, memory and timing of a fleet of clients that fight, as N grows (docs/rl/design.md#m2).

    python/.venv/bin/python python/bench/fleet.py --ns 2,8,16,32 [--seconds 30] [--settle 10]

For each N: a fresh local server (offline, a normal world with a fixed seed), N clients in scripted pairs with diamond swords
(they chase, strafe, jump, hit, die and respawn), one Python process stepping all of them every
tick (frame skip 1). After `settle` seconds, `seconds` of measurement:

- per client: CPU % of one core (all threads, from /proc), RSS and PSS (MB);
- `--infer-ms`: a pause per step standing for a policy's inference;
- timing: vector steps/s (20 = every tick), ticks per step (1 = none skipped), late ticks
  (a tick ran before its action arrived), the clients' own tick overruns (> 50 ms);
- the server's and the Python process's CPU, for scale (in real use the server is elsewhere).

Writes a JSON line per N to `--out` (default: stdout only).
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl.fleet import Fleet, FleetConfig  # noqa: E402
from horizon_rl.local import LocalServer, remove  # noqa: E402
from horizon_rl.scripted import Chaser  # noqa: E402

TICK = os.sysconf("SC_CLK_TCK")


def cpu_seconds(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    fields = stat[stat.rindex(")") + 2:].split()
    return (int(fields[11]) + int(fields[12])) / TICK


PAGE = os.sysconf("SC_PAGE_SIZE")


def rss_mb(pid):
    """Cheap (statm), so it can be sampled while the fleet steps."""
    try:
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * PAGE / 2**20
    except OSError:
        return None


def memory_mb(pid):
    """RSS and PSS from smaps_rollup: slow (it walks the page tables), so read once, after the
    timed window; read every few seconds during it, it stalls the stepping loop."""
    rss = pss = 0
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Rss:"):
                rss = int(line.split()[1])
            elif line.startswith("Pss:"):
                pss = int(line.split()[1])
    except OSError:
        return None
    return rss / 1024, pss / 1024


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def run(n, seconds, settle, workers, render_distance, infer_ms=0.0):
    root = Path(tempfile.mkdtemp(prefix=f"horizon-rl-bench-{n}-"))
    # A normal world: the stand-in server spends ~5 cores on a flat one with players (a Horizon
    # dedicated-server issue, not the clients'; docs/rl/design.md#m2-measurements).
    server = LocalServer(root, properties={"max-players": str(max(64, n + 4)), "level-seed": "1234",
                                           "level-type": "minecraft:normal"})
    server.command("gamerule spawn_radius 0")
    config = FleetConfig(server=server.address, n=n, pairs=True, workers=workers,
                         render_distance=render_distance, root=str(root / "fleet"))
    fleet = Fleet(config)
    try:
        t_start = time.monotonic()
        fleet.start(timeout=120 + n)
        started_in = time.monotonic() - t_start
        for i in range(n):
            server.command(f"give {config.name(i)} minecraft:diamond_sword")
        states, _ = fleet.reset(timeout=120 + n)
        states = list(states)
        policies = [Chaser(fleet.state_fields, seed=i, sword_slot=1) for i in range(n)]

        def play(duration, stats=None):
            end = time.monotonic() + duration
            steps = 0
            while time.monotonic() < end:
                actions = [p.act(s) for p, s in zip(policies, states)]
                if infer_ms:
                    time.sleep(infer_ms / 1000)  # a policy's inference
                new, infos, term, trunc, active = fleet.step(actions)
                steps += 1
                for i in range(n):
                    if not active[i]:
                        continue
                    states[i] = new[i]
                    info = infos[i]
                    if stats is not None and "ticks_elapsed" in info:
                        stats["slot_steps"] += 1
                        stats["ticks"] += info["ticks_elapsed"]
                        stats["late"] += info["late_ticks"]
                        stats["overrun_ms"] += info["tick_overrun_ms"]
                        stats["over_steps"] += info["tick_overrun_ms"] > 0
                        stats["hits"] += info["hit_landed"]
                        stats["deaths"] += info["died"]
                    if stats is not None:
                        stats["truncated"] += bool(trunc[i])
            return steps

        play(settle)
        pids = [s.proc.proc.pid for s in fleet.slots]
        cpu0 = [cpu_seconds(p) for p in pids]
        server_cpu0, py_cpu0 = cpu_seconds(server.proc.pid), time.process_time()
        stats = dict(slot_steps=0, ticks=0, late=0, overrun_ms=0.0, over_steps=0, hits=0, deaths=0,
                     truncated=0)
        rss, pss = [], []
        t0 = time.monotonic()
        steps = 0
        while time.monotonic() - t0 < seconds:
            steps += play(min(2.0, seconds - (time.monotonic() - t0)), stats)
            rss += [r for r in map(rss_mb, pids) if r is not None]
        dt = time.monotonic() - t0
        pss = [m[1] for m in map(memory_mb, pids) if m]
        cpu = [(c1 - c0) / dt * 100 for c0, c1 in zip(cpu0, (cpu_seconds(p) for p in pids))
               if c0 is not None and c1 is not None]
        restarts = sum(s.restarts for s in fleet.slots)
        ss = max(stats["slot_steps"], 1)
        return {
            "n": n,
            "infer_ms": infer_ms,
            "started_s": round(started_in, 1),
            "client_cpu_pct_mean": round(mean(cpu), 2),
            "client_cpu_pct_max": round(max(cpu), 2) if cpu else None,
            "clients_cpu_pct_total": round(sum(cpu), 1),
            "client_rss_mb_mean": round(mean(rss), 1),
            "client_rss_mb_max": round(max(rss), 1) if rss else None,
            "client_pss_mb_mean": round(mean(pss), 1),
            "server_cpu_pct": round((cpu_seconds(server.proc.pid) - server_cpu0) / dt * 100, 1),
            "python_cpu_pct": round((time.process_time() - py_cpu0) / dt * 100, 1),
            "vector_steps_per_s": round(steps / dt, 2),
            "ticks_per_step": round(stats["ticks"] / ss, 3),
            "late_ticks_per_1000_steps": round(1000 * stats["late"] / ss, 1),
            "overrun_steps_per_1000": round(1000 * stats["over_steps"] / ss, 1),
            "overrun_ms_per_1000_steps": round(1000 * stats["overrun_ms"] / ss, 1),
            "hits": stats["hits"],
            "deaths": stats["deaths"],
            "truncated": stats["truncated"],
            "restarts": restarts,
        }
    finally:
        fleet.close()
        server.stop()
        remove(root)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ns", default="2,8,16")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--settle", type=float, default=10)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--render-distance", type=int, default=2)
    ap.add_argument("--infer-ms", type=float, default=0.0,
                    help="sleep this long per step before sending, as a policy's inference would")
    ap.add_argument("--out")
    args = ap.parse_args()
    for n in (int(x) for x in args.ns.split(",")):
        result = run(n, args.seconds, args.settle, args.workers, args.render_distance, args.infer_ms)
        line = json.dumps(result)
        print(line, flush=True)
        if args.out:
            with open(args.out, "a") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
