#!/usr/bin/env python3
"""The login slot end to end: a pair of clients in one process (model B, so a sibling's options
carry it too) on a local server, each told `login_slot=1` and given a sword there. What the
server holds shows in the opponent's view of the main hand. Needs the fast build and .local/.

    python/.venv/bin/python python/tests/e2e_login_slot.py

Checks: both hold the sword once joined; a step's slot replaces it; a kicked client that rejoins
holds the sword again for the server too, though the server remembered slot 6 for it (slot 1
is the slot a client's first tick sends before the server's `SetHeldSlot`); the other keeps 6.
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
    config = FleetConfig(server=server.address, n=2, pairs=True, login_slot=1, clients_per_process=2,
                         root=str(root / "fleet"))
    fleet = Fleet(config)
    fields = sword = None

    def selected(states):
        return [int(round(s[fields["self_selected"]][0])) for s in states]

    def sword_seen(states):
        """Whether each slot's opponent sees it holding the sword (the server's view)."""
        hands = [int(round(s[fields["target_main_hand"]][0])) for s in states]
        return [hands[1] == sword, hands[0] == sword]

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
            server.command(f"give {config.name(i)} minecraft:diamond_sword")
        fleet.reset()
        fields = fleet.state_fields
        sword = fleet.items.index("minecraft:diamond_sword") + 1  # ids are registry id + 1
        states = play(10, until=lambda st, _: sword_seen(st) == [True, True])
        check(selected(states) == [0, 0] and sword_seen(states) == [True, True],
              f"both hold the sword in slot 1 after login ({selected(states)}, {sword_seen(states)})")

        states = play(1, hotbar=6)
        check(selected(states) == [5, 5], f"a step's slot replaces it ({selected(states)})")
        # Let the server hear of slot 6 before it saves the kicked player.
        play(1)

        server.command(f"kick {config.name(0)} testing")
        states = play(60, until=lambda st, started: started[0] and sword_seen(st)[0])
        check(selected(states) == [0, 5] and sword_seen(states) == [True, False],
              f"the rejoined client holds the sword again for the server, the other keeps 6 "
              f"({selected(states)}, {sword_seen(states)})")
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
