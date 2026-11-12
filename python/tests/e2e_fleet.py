#!/usr/bin/env python3
"""M2 end to end: a fleet of 4 clients in fighting pairs on a local server, and the faults it must
survive, each hitting one slot while the others play on. Needs the fast build and .local/.

    python/.venv/bin/python python/tests/e2e_fleet.py

Checks: all join (staggered) and fight; a killed process comes back (`crash`); a kicked client
joins again by itself (`disconnected`); a frozen process is found by its heartbeat and replaced
(`hang`); a death ends the episode (`terminated`) and the slot starts the next one; no fault
truncates another slot.
"""
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl.fleet import Fleet, FleetConfig  # noqa: E402
from horizon_rl.local import LocalServer, remove  # noqa: E402
from horizon_rl.scripted import Chaser  # noqa: E402

failures = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


class Run:
    def __init__(self, fleet):
        self.fleet = fleet
        self.policies = [Chaser(fleet.state_fields, seed=i, sword_slot=1) for i in range(len(fleet.slots))]
        self.states = None
        self.ended = {}  # slot -> list of (kind, reason)
        self.started = {}  # slot -> new episodes seen
        self.hits = 0
        self.steps = 0

    def play(self, seconds, until=None):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            actions = [p.act(s) for p, s in zip(self.policies, self.states)]
            states, infos, term, trunc, active = self.fleet.step(actions)
            self.steps += 1
            for i in range(len(states)):
                if not active[i]:
                    continue
                self.states[i] = states[i]
                info = infos[i]
                self.hits += int(info.get("hit_landed", 0))
                if info.get("episode_start"):
                    self.started[i] = self.started.get(i, 0) + 1
                    if info.get("reset_reason"):
                        self.ended.setdefault(i, []).append(("between", info["reset_reason"]))
                if term[i]:
                    self.ended.setdefault(i, []).append(("terminated", "died"))
                if trunc[i]:
                    self.ended.setdefault(i, []).append(("truncated", info.get("reset_reason", "")))
            if until is not None and until():
                return True
        return False


def main():
    root = Path(tempfile.mkdtemp(prefix="horizon-rl-fleet-e2e-"))
    server = LocalServer(root)
    config = FleetConfig(server=server.address, n=4, pairs=True, root=str(root / "fleet"))
    fleet = Fleet(config)
    try:
        t0 = time.monotonic()
        fleet.start()
        for i in range(4):
            server.wait_joined(config.name(i))
            server.command(f"give {config.name(i)} minecraft:diamond_sword")
        check(True, f"4 clients started and joined in {time.monotonic() - t0:.1f} s")
        states, infos = fleet.reset()
        run = Run(fleet)
        run.states = list(states)
        check(all(info["alive"] == 1 for info in infos), "all alive after reset")
        seen = set()

        def all_seen():
            # Each slot's partner shows up once the server has sent it (a few ticks after join).
            for i, slot in enumerate(fleet.slots):
                if slot.last and slot.last[1].get("target_present") == 1:
                    seen.add(i)
            return len(seen) == 4
        check(run.play(10, until=all_seen), f"each sees its partner ({sorted(seen)})")
        run.play(6)
        check(run.hits > 0, f"they fight ({run.hits} hits in {run.steps} steps)")

        def came_back(i, kind):
            before = run.started.get(i, 0)
            # Truncated mid-episode, or named at the next start when it hit between episodes.
            return lambda: run.started.get(i, 0) > before and any(
                k in ("truncated", "between") and r.startswith(kind) for k, r in run.ended.get(i, []))

        # A crash: kill slot 0's process.
        run.ended.clear()
        fleet.kill(0)
        back = run.play(60, until=came_back(0, "crash"))
        check(back, f"a killed process is restarted and plays again ({run.ended.get(0)})")
        check(fleet.slots[0].restarts == 1, f"one restart ({fleet.slots[0].restarts})")

        # A kick: the client joins again by itself, no restart.
        run.ended.clear()
        server.command(f"kick {config.name(1)} testing")
        back = run.play(60, until=came_back(1, "disconnected"))
        check(back, f"a kicked client rejoins by itself ({run.ended.get(1)})")
        check(fleet.slots[1].restarts == 0, "without a restart")

        # A hang: freeze slot 2; its heartbeat stops and the supervisor replaces it.
        run.ended.clear()
        os.kill(fleet.slots[2].proc.proc.pid, signal.SIGSTOP)
        frozen = time.monotonic()
        back = run.play(60, until=came_back(2, "hang"))
        check(back, f"a frozen process is replaced ({run.ended.get(2)})")
        check(time.monotonic() - frozen < 30, f"within 30 s ({time.monotonic() - frozen:.1f} s)")

        # A death ends the episode and the slot starts the next one.
        # (Wait until it is mid-episode, not dead from the fight already.)
        run.play(20, until=lambda: fleet.slots[3].state == "ready" and not fleet.slots[3].episode_start)
        run.ended.clear()
        before = run.started.get(3, 0)
        server.command(f"kill {config.name(3)}")
        run.play(20, until=lambda: run.started.get(3, 0) > before)
        check(("terminated", "died") in run.ended.get(3, []), f"a death terminates ({run.ended.get(3)})")
        check(run.started.get(3, 0) > before, "and a new episode starts")

        others = {i: e for i, e in run.ended.items() if i != 3 and any(k == "truncated" for k, _ in e)}
        check(not others, f"no other slot was truncated by it ({others})")
        print("stats:", fleet.stats())
    finally:
        fleet.close(keep_files=True)
        server.stop()
    if failures:
        print(f"\n{len(failures)} failed; logs kept in {root}")
        sys.exit(1)
    print("\nall passed")
    remove(root)


main()
