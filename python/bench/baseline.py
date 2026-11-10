#!/usr/bin/env python3
"""Baseline: CPU and RSS of horizon-client joined to a local offline dedicated server
(docs/rl/design.md#baseline-measurements).

    python3 python/bench/baseline.py . [--seconds 40] [--settle 15] [--fleet N]

Every game folder and XDG/HORIZON home is a fresh temp dir (tools/lab.game_env)."""
import argparse, json, os, socket, subprocess, sys, tempfile, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("repo")
ap.add_argument("--seconds", type=float, default=45)
ap.add_argument("--settle", type=float, default=15)
ap.add_argument("--only", help="comma list of config names")
ap.add_argument("--fleet", type=int, default=0, help="run N headless-rd2 clients at once instead")
args = ap.parse_args()
REPO = Path(args.repo).resolve()
sys.path.insert(0, str(REPO / "tools"))
import lab  # noqa: E402

BIN = REPO / "target/fast/horizon-client"
RES = REPO / ".local/horizon-client/26.3"
TICK = os.sysconf("SC_CLK_TCK")
PAGE = os.sysconf("SC_PAGE_SIZE")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(path, text, timeout, proc=None):
    end = time.time() + timeout
    while time.time() < end:
        if path.exists() and text in path.read_text(errors="replace"):
            return True
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"process exited ({proc.returncode}); see {path}")
        time.sleep(0.25)
    raise TimeoutError(f"{text!r} not in {path} after {timeout}s")


def proc_sample(pid):
    stat = Path(f"/proc/{pid}/stat").read_text()
    fields = stat[stat.rindex(")") + 2:].split()
    cpu = (int(fields[11]) + int(fields[12])) / TICK  # utime + stime, all threads
    status = {l.split(":")[0]: l.split(":")[1].strip() for l in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in l}
    rss_kb = int(status["VmRSS"].split()[0])
    threads = int(status["Threads"])
    pss_kb = 0
    try:
        for l in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if l.startswith("Pss:"):
                pss_kb = int(l.split()[1])
    except OSError:
        pass
    return cpu, rss_kb, pss_kb, threads


def measure(pid, seconds):
    cpu0, *_ = proc_sample(pid)
    t0 = time.time()
    rss, pss, thr = [], [], []
    while time.time() - t0 < seconds:
        time.sleep(1)
        _, r, p, t = proc_sample(pid)
        rss.append(r); pss.append(p); thr.append(t)
    cpu1, *_ = proc_sample(pid)
    dt = time.time() - t0
    return {
        "cpu_pct_one_core": round(100 * (cpu1 - cpu0) / dt, 1),
        "rss_mb_mean": round(sum(rss) / len(rss) / 1024, 1),
        "rss_mb_max": round(max(rss) / 1024, 1),
        "pss_mb_mean": round(sum(pss) / len(pss) / 1024, 1),
        "threads": max(thr),
    }


def start_server(root, env):
    sdir = root / "server"
    sdir.mkdir()
    port = free_port()
    props = {"online-mode": "false", "server-ip": "127.0.0.1", "server-port": port, "white-list": "false",
             "level-name": "world", "level-seed": "1234", "spawn-protection": "0", "view-distance": "10",
             "simulation-distance": "10", "gamemode": "survival", "difficulty": "peaceful"}
    (sdir / "server.properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()))
    (sdir / "eula.txt").write_text("eula=true\n")
    log = sdir / "out.log"
    proc = subprocess.Popen([str(BIN), "--resources", str(RES), "--dedicated", "--nogui"], cwd=sdir, env=env,
                            stdin=subprocess.PIPE, stdout=log.open("w"), stderr=subprocess.STDOUT, text=True)
    wait_for(log, "Done (", 300, proc)
    return proc, port, log


CONFIGS = {
    "headless-rd2": ["--headless", "--render-distance", "2"],
    "headless-rd10": ["--headless", "--render-distance", "10"],
    "offscreen-rd2-128": ["--offscreen", "--render-distance", "2", "--width", "128", "--height", "128"],
    "offscreen-rd10-default": ["--offscreen", "--render-distance", "10"],
}


def main():
    root = Path(tempfile.mkdtemp(prefix="rl-baseline-"))
    env = lab.game_env(HORIZON_TELEMETRY_URL="off")
    server, port, slog = start_server(root, env)
    results = {}
    if args.fleet:
        procs = []
        try:
            for i in range(args.fleet):
                gdir = root / f"fleet-{i}"; gdir.mkdir()
                cenv = lab.game_env(dict(os.environ), HORIZON_TELEMETRY_URL="off")
                cmd = [str(BIN), "--resources", str(RES), "--game-dir", str(gdir), "--no-pause",
                       "--server", f"127.0.0.1:{port}", "--name", f"Fleet{i}", *CONFIGS["headless-rd2"]]
                procs.append(subprocess.Popen(cmd, cwd=gdir, env=cenv, stdout=(gdir / "out.log").open("w"), stderr=subprocess.STDOUT))
            for i, p in enumerate(procs):
                wait_for(slog, f"Fleet{i} joined the game", 180, p)
            time.sleep(args.settle)
            import threading
            per = [None] * len(procs)
            ts = [threading.Thread(target=lambda i=i, p=p: per.__setitem__(i, measure(p.pid, args.seconds))) for i, p in enumerate(procs)]
            srv = {}
            st = threading.Thread(target=lambda: srv.update(measure(server.pid, args.seconds)))
            [t.start() for t in ts]; st.start(); [t.join() for t in ts]; st.join()
            tot_cpu = sum(r["cpu_pct_one_core"] for r in per)
            print(json.dumps({"n": len(procs), "alive": sum(p.poll() is None for p in procs),
                              "cpu_pct_total": round(tot_cpu, 1), "cpu_pct_mean": round(tot_cpu / len(per), 1),
                              "rss_mb_mean": round(sum(r["rss_mb_mean"] for r in per) / len(per), 1),
                              "pss_mb_mean": round(sum(r["pss_mb_mean"] for r in per) / len(per), 1),
                              "server": srv}))
        finally:
            for p in procs: p.terminate()
            for p in procs:
                try: p.wait(15)
                except subprocess.TimeoutExpired: p.kill()
            server.stdin.write("stop\n"); server.stdin.flush(); server.wait(30)
        return
    try:
        names = args.only.split(",") if args.only else list(CONFIGS)
        for i, name in enumerate(names):
            gdir = root / f"client-{name}"
            gdir.mkdir()
            cenv = lab.game_env(dict(os.environ), HORIZON_TELEMETRY_URL="off")  # its own fresh homes
            player = f"Base{i}"
            clog = gdir / "out.log"
            cmd = [str(BIN), "--resources", str(RES), "--game-dir", str(gdir), "--no-pause",
                   "--server", f"127.0.0.1:{port}", "--name", player, *CONFIGS[name]]
            started = time.time()
            proc = subprocess.Popen(cmd, cwd=gdir, env=cenv, stdout=clog.open("w"), stderr=subprocess.STDOUT)
            try:
                wait_for(slog, f"{player} joined the game", 180, proc)
                joined = time.time() - started
                time.sleep(args.settle)
                r = measure(proc.pid, args.seconds)
                r["join_s"] = round(joined, 1)
                r["alive"] = proc.poll() is None
                results[name] = r
                print(name, json.dumps(r), flush=True)
            except Exception as e:  # keep measuring the others
                results[name] = {"error": str(e)}
                print(name, "ERROR", e, flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(15)
                except subprocess.TimeoutExpired:
                    proc.kill()
    finally:
        try:
            server.stdin.write("stop\n"); server.stdin.flush(); server.wait(30)
        except Exception:
            server.kill()
    out = root / "baseline.json"
    out.write_text(json.dumps(results, indent=2))
    print("work dir:", root)


main()
