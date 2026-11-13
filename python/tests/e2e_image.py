#!/usr/bin/env python3
"""M3 end to end: grayscale image observations from a headless client (no display), on a local
offline flat-world server. Needs the fast build, the prepared resources (.local/) and a GPU.

    python/.venv/bin/python python/tests/e2e_image.py

Checks: the WELCOME's frame slot has the asked size; each observation's frame is drawn for it
(`frame_fresh`) and is a real picture (not blank); looking up shows the bright sky, looking down
the darker ground; the opponent, walked up to with the observation only, darkens the sky above
the horizon where it stands, and is gone when the agent turns away; with frame skip 4 a frame
comes with every observation (one per 4 ticks); the HUD is off by default and `--rl-hud` draws
it (the hotbar changes the bottom rows); a state-only client opens no renderer.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

from horizon_rl import HorizonClient, proto  # noqa: E402
from horizon_rl.local import RENDER_OPTIONS, ClientProcess, LocalServer, remove  # noqa: E402

failures = []
W, H = 96, 64


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what, flush=True)
    if not cond:
        failures.append(what)


def image_client(server, name, root, extra=()):
    # As the fleet starts them: without the HUD, the atlas holds only what the world draws (the
    # opponent seen below is drawn from it).
    env = {"HORIZON_SUBMIT_THREAD": "0"}
    if "--rl-hud" not in extra:
        env["HORIZON_LEAN_ATLAS"] = "1"
    return ClientProcess(server.address, name, root, extra=["--rl-image", f"{W}x{H}", *extra],
                         render_options=RENDER_OPTIONS, env=env)


def sky_band(frame):
    """The rows just above the horizon (pitch 0 puts it mid-frame), middle third of the columns."""
    return frame[H // 2 - 12:H // 2 - 1, W // 3:2 * W // 3].astype(float)


def main():
    root = Path(tempfile.mkdtemp(prefix="horizon-rl-e2e-image-"))
    server = LocalServer(root)
    procs = []
    try:
        agent = image_client(server, "Agent", root)
        procs.append(agent)
        server.wait_joined("Agent")
        dummy = ClientProcess(server.address, "Dummy", root, rl=False)
        procs.append(dummy)
        server.wait_joined("Dummy")
        with HorizonClient(agent.socket) as env:
            check(env.frame_shape == (H, W), f"the frame slot is {H}x{W} ({env.frame_shape})")
            state, info = env.reset()
            f = lambda name: env.field(state, name)  # noqa: E731
            state, info = env.step(yaw=0.0, pitch=0.0)
            frame = env.frame()
            check(info["frame_fresh"] == 1, "the frame is drawn for the observation")
            check(frame.std() > 5, f"the frame is a picture (std {frame.std():.1f})")
            check(info["frame_ms"] > 0, f"its cost is reported ({info['frame_ms']:.2f} ms)")

            for _ in range(3):
                state, info = env.step(pitch=-90.0)
            sky = env.frame().mean()
            for _ in range(3):
                state, info = env.step(pitch=90.0)
            ground = env.frame().mean()
            check(sky > ground + 30, f"the sky is brighter than the ground ({sky:.0f} vs {ground:.0f})")

            # Walk up to the dummy with the observation only, then look at it level.
            for _ in range(200):
                yaw = f("self_yaw")[0] + f("target_bearing")[0]
                near = f("target_distance")[0] < 2.5
                state, info = env.step(0 if near else proto.FORWARD, yaw=yaw, pitch=0.0)
                if near:
                    break
            check(f("target_distance")[0] < 3, f"walked up to the opponent ({f('target_distance')[0]:.1f})")
            for _ in range(5):
                yaw = f("self_yaw")[0] + f("target_bearing")[0]
                state, info = env.step(0, yaw=yaw, pitch=0.0)
            facing = sky_band(env.frame())
            for _ in range(5):
                state, info = env.step(0, yaw=yaw + 180.0, pitch=0.0)
            away = sky_band(env.frame())
            # The sky band is plain sky, bright all over, unless the opponent stands in it. (Not
            # "even": the sun is in it when the agent happens to face that way.)
            check(away.mean() > 120 and away.min() > 140,
                  f"facing away, the band above the horizon is plain sky ({away.mean():.0f}, darkest {away.min():.0f})")
            check(away.mean() - facing.mean() > 20,
                  f"facing the opponent, it darkens that band ({facing.mean():.0f} vs {away.mean():.0f})")

            # Frame skip 4: one frame per observation, after the 4 ticks.
            for _ in range(5):
                state, info = env.step(0, frame_skip=4)
                check(info["ticks_elapsed"] == 4 and info["frame_fresh"] == 1,
                      f"frame skip 4: 4 ticks and a fresh frame ({info['ticks_elapsed']}, {info['frame_fresh']})")
            bottom_plain = env.frame()[-12:].astype(float)

        # The HUD: another client with --rl-hud, looking the same way at the same place.
        hud = image_client(server, "Hud", root, extra=["--rl-hud"])
        procs.append(hud)
        server.wait_joined("Hud")
        server.command("tp Hud Agent")
        with HorizonClient(hud.socket) as env:
            state, info = env.reset()
            for _ in range(10):
                state, info = env.step(yaw=yaw + 180.0, pitch=0.0)
            bottom_hud = env.frame()[-12:].astype(float)
        diff = np.abs(bottom_hud - bottom_plain).mean()
        check(diff > 10, f"--rl-hud draws the hotbar over the bottom rows (mean difference {diff:.1f})")

        # A state-only client never builds a renderer.
        log = (root / "Dummy" / "out.log").read_text(errors="replace")
        check("Horizon renderer:" not in log, "a client without --rl-image opens no renderer")
        log = (root / "Agent" / "out.log").read_text(errors="replace")
        check("Headless renderer: 96x64 offscreen" in log, "an image client renders headless at its size")
    finally:
        for p in procs:
            p.stop()
        server.stop()
        remove(root)
    if failures:
        print(f"{len(failures)} failed")
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
