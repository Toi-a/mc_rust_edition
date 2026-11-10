#!/usr/bin/env python3
"""M1 end to end: one Python-driven headless client against a second player, on a local offline
flat-world server. Needs the fast build and the prepared resources (.local/).

    python/.venv/bin/python python/tests/e2e_duel.py

Checks: join and observe; walking and turning change the observation as asked; the opponent is in
the observation without its health, and found with only the observation (bearing, distance);
attacks land (`info.hit_landed`) and its health falls in `info.target_health`; a killed opponent
counts in `kills_total`; the RL player's own death and respawn show in `info` and `reset()`.
"""
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl import HorizonClient, proto  # noqa: E402
from horizon_rl.local import ClientProcess, LocalServer, remove  # noqa: E402

failures = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def main():
    root = Path(tempfile.mkdtemp(prefix="horizon-rl-e2e-"))
    server = LocalServer(root)
    procs = []
    try:
        agent = ClientProcess(server.address, "Agent", root)
        procs.append(agent)
        server.wait_joined("Agent")
        dummy = ClientProcess(server.address, "Dummy", root, rl=False)
        procs.append(dummy)
        server.wait_joined("Dummy")
        # Give both something to hit with and the agent a world to see the other in.
        server.command("give Agent minecraft:diamond_sword")
        server.command("gamerule spawn_radius 0")
        with HorizonClient(agent.socket) as env:
            state, info = env.reset()
            f = lambda name: env.field(state, name)  # noqa: E731
            check(info["alive"] == 1 and info["connected"] == 1, "reset: alive and connected")
            check(f("self_health")[0] == 20.0, f"own health is in the observation ({f('self_health')})")
            check("target_health" not in env.state_fields, "no target health in the state layout")

            # Facing: absolute yaw and pitch.
            state, info = env.step(yaw=90.0, pitch=10.0)
            check(abs(f("self_yaw")[0] - 90.0) < 1e-3 and abs(f("self_pitch")[0] - 10.0) < 1e-3,
                  f"look is absolute ({f('self_yaw')[0]}, {f('self_pitch')[0]})")
            check(info["ticks_elapsed"] == 1, f"one tick per step ({info['ticks_elapsed']})")

            # Walk: forward for 20 ticks moves the player toward -x (yaw 90 faces west).
            x0 = f("self_pos")[0]
            for _ in range(20):
                state, info = env.step(proto.FORWARD)
            check(f("self_pos")[0] < x0 - 1.5, f"forward walks west ({x0:.2f} -> {f('self_pos')[0]:.2f})")
            state, info = env.step(0)
            check(info["late_ticks"] == 0, f"no late ticks while Python keeps up ({info['late_ticks']})")

            # A stall: ticks run on without a new action, and say so.
            time.sleep(0.5)
            state, info = env.step(0)
            check(info["late_ticks"] >= 5, f"a 0.5 s stall shows as late ticks ({info['late_ticks']})")

            # Find the dummy with only the observation, and fight it.
            seen = [env.step(0) for _ in range(5)][-1]
            state, info = seen
            check(f("target_present")[0] == 1.0 and info["target_present"] == 1, "the opponent is observed")
            check(math.isfinite(info["target_health"]), f"its health is in info ({info['target_health']})")
            sword = env.items.index("minecraft:diamond_sword") + 1
            slot = list(f("self_hotbar_item")).index(sword) + 1 if sword in f("self_hotbar_item") else 1
            state, info = env.step(0, hotbar=slot)
            hits = landed = 0
            kills0 = info["kills_total"]
            health_start = info["target_health"]
            low = health_start
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                yaw = f("self_yaw")[0] + f("target_bearing")[0]
                pitch = f("self_pitch")[0] + f("target_bearing")[1]
                dist = f("target_distance")[0]
                buttons = proto.FORWARD | proto.SPRINT if dist > 2.5 else 0
                if dist < 3.0 and f("self_attack_cooldown")[0] >= 1.0:
                    buttons |= proto.ATTACK
                    hits += 1
                state, info = env.step(buttons, yaw=yaw, pitch=pitch)
                if "-v" in sys.argv and info["tick"] % 20 == 0:
                    print("  dist", round(float(dist), 2), "cooldown", f("self_attack_cooldown")[0],
                          "held", f("self_hotbar_item")[int(f("self_selected")[0])], "target hp", info["target_health"])
                landed += info["hit_landed"]
                if math.isfinite(info["target_health"]):
                    low = min(low, info["target_health"])
                if info["kills_total"] > kills0:
                    break
            check(landed > 0, f"attacks land ({landed} of {hits})")
            check(low < health_start, f"its health falls in info ({health_start} -> {low})")
            check(info["kills_total"] == kills0 + 1, f"the kill is counted ({info['kills_total']})")

            # Our own death: /kill, then the respawn the client clicks itself.
            server.command("kill Agent")
            died = 0
            for _ in range(60):
                state, info = env.step(0)
                died += info["died"]
                if died:
                    break
            check(died == 1 and info["deaths_total"] == 1, f"death is reported ({died}, {info['deaths_total']})")
            state, info = env.reset()
            check(info["alive"] == 1 and f("self_health")[0] == 20.0, "reset() returns after the respawn")
    finally:
        for p in procs:
            p.stop()
        server.stop()
    if failures:
        print(f"\n{len(failures)} failed; logs kept in {root}")
        sys.exit(1)
    print("\nall passed")
    remove(root)


main()
