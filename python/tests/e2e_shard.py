#!/usr/bin/env python3
"""Process model B end to end: 4 image clients in 2 processes (`clients_per_process=2`,
`--rl-client`), in fighting pairs on a local flat-world server. Needs the fast build, .local/ and
a GPU.

    python/.venv/bin/python python/tests/e2e_shard.py

Checks: 4 slots run in 2 processes, one GPU device each; all join, see their partner and fight;
each client draws its own view (fresh frames, real pictures, not each other's); a kicked client
of a shard rejoins by itself; a killed process restarts both its clients (`crash` and
`crash (shard)`) and leaves the other process's alone.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from e2e_fleet import Run  # noqa: E402
from horizon_rl.fleet import Fleet, FleetConfig  # noqa: E402
from horizon_rl.local import LocalServer, remove  # noqa: E402

failures = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def main():
    root = Path(tempfile.mkdtemp(prefix="horizon-rl-shard-e2e-"))
    server = LocalServer(root)
    config = FleetConfig(server=server.address, n=4, pairs=True, obs="both", image_size=(64, 64),
                         clients_per_process=2, root=str(root / "fleet"))
    fleet = Fleet(config)
    try:
        t0 = time.monotonic()
        fleet.start()
        for i in range(4):
            server.wait_joined(config.name(i))
            server.command(f"give {config.name(i)} minecraft:diamond_sword")
        pids = {s.proc.proc.pid for s in fleet.slots}
        check(len(pids) == 2, f"4 clients in {len(pids)} processes, started in {time.monotonic() - t0:.1f} s")
        log = (root / "fleet" / config.name(0) / "out.log").read_text(errors="replace")
        check("RL shard: 2 clients in this process" in log, "the first process runs 2 clients")
        devices = log.count("requesting the device")
        check(devices == 1, f"on one GPU device ({devices} opened)")

        states, infos = fleet.reset()
        run = Run(fleet)
        run.states = list(states)
        check(all(info["alive"] == 1 for info in infos), "all alive after reset")
        seen = set()

        def all_seen():
            for i, slot in enumerate(fleet.slots):
                if slot.last and slot.last[1].get("target_present") == 1:
                    seen.add(i)
            return len(seen) == 4
        check(run.play(10, until=all_seen), f"each sees its partner ({sorted(seen)})")
        run.play(6)
        check(run.hits > 0, f"they fight ({run.hits} hits in {run.steps} steps)")

        fresh = [s.last[1].get("frame_fresh") for s in fleet.slots]
        check(all(f == 1 for f in fresh), f"every client's frame is fresh ({fresh})")
        frames = fleet.frames.astype(float)
        stds = [round(f.std(), 1) for f in frames]
        check(all(sd > 5 for sd in stds), f"every frame is a picture (std {stds})")
        diff = np.abs(frames[0] - frames[1]).mean()
        check(diff > 1, f"the two clients of a process draw their own views (mean difference {diff:.1f})")

        def came_back(i, kind):
            before = run.started.get(i, 0)
            return lambda: run.started.get(i, 0) > before and any(
                k in ("truncated", "between") and r.startswith(kind) for k, r in run.ended.get(i, []))

        # A kick of the second client of a process: it joins again by itself, no restart.
        run.ended.clear()
        server.command(f"kick {config.name(1)} testing")
        back = run.play(60, until=came_back(1, "disconnected"))
        check(back, f"a kicked client of a shard rejoins by itself ({run.ended.get(1)})")
        check(fleet.slots[1].restarts == 0 and fleet.slots[0].restarts == 0, "without a restart")

        # A crash: kill the first process; both its clients restart, the others play on.
        run.ended.clear()
        fleet.kill(0)
        first, second = came_back(0, "crash"), came_back(1, "crash")
        back = run.play(90, until=lambda: first() and second())
        check(back, f"a killed process restarts both its clients ({run.ended.get(0)}, {run.ended.get(1)})")
        reasons = [r for i in (0, 1) for k, r in run.ended.get(i, []) if k != "terminated"]
        check(any(r.endswith("(shard)") for r in reasons), f"one of them names the shard ({reasons})")
        check([s.restarts for s in fleet.slots] == [1, 1, 0, 0],
              f"one restart each for that process's slots ({[s.restarts for s in fleet.slots]})")
        others = {i: e for i, e in run.ended.items() if i >= 2 and any(k == "truncated" for k, _ in e)}
        check(not others, f"the other process's clients were not truncated ({others})")
        print("stats:", fleet.stats())
    finally:
        fleet.close(keep_files=True)
        server.stop()
    if failures:
        print(f"\n{len(failures)} failed; logs kept in {root}")
        sys.exit(1)
    print("\nall passed")
    remove(root)


if __name__ == "__main__":
    main()
