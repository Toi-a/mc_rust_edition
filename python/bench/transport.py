#!/usr/bin/env python3
"""Transport benchmark (docs/rl/design.md#transport): the binary RL socket + shared memory against
the JSON control socket (`horizon-ctl`'s `state` + `entities`), on one headless client joined to a
local server with a second player around.

    python/.venv/bin/python python/bench/transport.py [--seconds 30]

For each transport, over the same time at the 20 Hz tick: observations per second, Python CPU
per observation, the game's CPU %, bytes per observation, and the wait from request to answer.
"""
import argparse
import json
import os
import socket
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl import HorizonClient  # noqa: E402
from horizon_rl.local import ClientProcess, LocalServer, remove  # noqa: E402

TICK = os.sysconf("SC_CLK_TCK")


def cpu_seconds(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat[stat.rindex(")") + 2:].split()
    return (int(fields[11]) + int(fields[12])) / TICK


class Ctl:
    def __init__(self, path):
        deadline = time.monotonic() + 60
        while True:
            try:
                self.s = socket.socket(socket.AF_UNIX)
                self.s.connect(str(path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                self.s.close()
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        self.buf = b""

    def __call__(self, line):
        self.s.sendall(line.encode() + b"\n")
        while b"\n" not in self.buf:
            chunk = self.s.recv(1 << 20)
            if not chunk:
                raise ConnectionError("closed")
            self.buf += chunk
        out, self.buf = self.buf.split(b"\n", 1)
        return out


def run_binary(sock, pid, seconds):
    env = HorizonClient(sock)
    env.reset()
    waits, n = [], 0
    cpu0, py0, t0 = cpu_seconds(pid), time.process_time(), time.monotonic()
    while time.monotonic() - t0 < seconds:
        a = time.perf_counter()
        state, info = env.step(0)
        waits.append(time.perf_counter() - a)
        n += 1
    dt = time.monotonic() - t0
    out = {"obs_per_s": n / dt, "python_cpu_us_per_obs": 1e6 * (time.process_time() - py0) / n,
           "game_cpu_pct": 100 * (cpu_seconds(pid) - cpu0) / dt,
           "bytes_per_obs": 4 * (len(state) + len(env.info_fields)) + 8 + 16 + 28,
           "wait_ms_median": 1000 * statistics.median(waits)}
    env.close()
    return out


def run_json(path, pid, seconds):
    ctl = Ctl(path)
    waits, sizes, n = [], [], 0
    cpu0, py0, t0 = cpu_seconds(pid), time.process_time(), time.monotonic()
    next_at = t0
    while time.monotonic() - t0 < seconds:
        # One observation per tick, as the RL loop asks for.
        next_at += 0.05
        a = time.perf_counter()
        state = ctl("state")
        ents = ctl("entities 32")
        json.loads(state)
        json.loads(ents)
        waits.append(time.perf_counter() - a)
        sizes.append(len(state) + len(ents))
        n += 1
        pause = next_at - time.monotonic()
        if pause > 0:
            time.sleep(pause)
    dt = time.monotonic() - t0
    return {"obs_per_s": n / dt, "python_cpu_us_per_obs": 1e6 * (time.process_time() - py0) / n,
            "game_cpu_pct": 100 * (cpu_seconds(pid) - cpu0) / dt,
            "bytes_per_obs": statistics.mean(sizes), "wait_ms_median": 1000 * statistics.median(waits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30)
    args = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix="horizon-rl-transport-"))
    server = LocalServer(root)
    procs = []
    try:
        ctl_path = root / "ctl.sock"
        agent = ClientProcess(server.address, "Agent", root, extra=["--control-socket", str(ctl_path)])
        procs.append(agent)
        server.wait_joined("Agent")
        procs.append(ClientProcess(server.address, "Dummy", root, rl=False))
        server.wait_joined("Dummy")
        time.sleep(5)
        idle0, t0 = cpu_seconds(agent.proc.pid), time.monotonic()
        time.sleep(args.seconds / 2)
        idle = 100 * (cpu_seconds(agent.proc.pid) - idle0) / (time.monotonic() - t0)
        results = {"idle_game_cpu_pct": idle,
                   "binary": run_binary(agent.socket, agent.proc.pid, args.seconds),
                   "json": run_json(ctl_path, agent.proc.pid, args.seconds)}
        print(json.dumps(results, indent=2))
    finally:
        for p in procs:
            p.stop()
        server.stop()
        remove(root)


main()
