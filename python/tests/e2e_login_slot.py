#!/usr/bin/env python3
"""The login slot end to end: two clients in one process (model B, so a sibling's options carry
it too) on a local server, each told `hotbar=3`. Needs the fast build and .local/.

    python/.venv/bin/python python/tests/e2e_login_slot.py

Checks: both hold slot 3 once joined; a step's slot replaces it; a kicked client that rejoins
holds slot 3 again, though the server remembered the other slot; the other keeps the step's.
"""
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl.fleet import Fleet, FleetConfig  # noqa: E402
from horizon_rl.local import LocalServer  # noqa: E402

failures = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def main():
    root = Path(tempfile.mkdtemp(prefix="horizon-rl-slot-e2e-"))
    server = LocalServer(root)
    config = FleetConfig(server=server.address, n=2, hotbar=3, clients_per_process=2,
                         root=str(root / "fleet"))
    fleet = Fleet(config)
    field = None

    def selected(states):
        return [int(round(s[field][0])) for s in states]

    def play(seconds, hotbar=0, until=None):
        deadline = time.monotonic() + seconds
        states = None
        started = [False, False]
        while time.monotonic() < deadline:
            states, infos, _, _, active = fleet.step([(0, hotbar, math.nan, math.nan)] * 2)
            for i, info in enumerate(infos):
                started[i] |= bool(active[i] and info.get("episode_start"))
            if until is not None and until(states, started):
                break
        return states

    try:
        fleet.start()
        for i in range(2):
            server.wait_joined(config.name(i))
        states, _ = fleet.reset()
        field = fleet.state_fields["self_selected"]
        states = play(2)
        check(selected(states) == [2, 2], f"both hold slot 3 after login ({selected(states)})")

        states = play(1, hotbar=6)
        check(selected(states) == [5, 5], f"a step's slot replaces it ({selected(states)})")
        # Let the server hear of slot 6 before it saves the kicked player.
        play(1)

        server.command(f"kick {config.name(0)} testing")
        states = play(60, until=lambda st, started: started[0] and selected(st)[0] == 2)
        check(selected(states) == [2, 5],
              f"the rejoined client holds slot 3 again, the other keeps 6 ({selected(states)})")
        check(fleet.slots[0].restarts == 0, "without a restart")
    finally:
        fleet.close(keep_files=True)
        server.stop()
    if failures:
        print(f"{len(failures)} failed")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
