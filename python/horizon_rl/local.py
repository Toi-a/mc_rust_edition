"""Local stand-ins for testing: the repo's own dedicated server in offline mode, and client
processes, each in fresh scratch folders (never a real player's: CLAUDE.md, "Keep your runs away
from real player data")."""
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path


def repo_root():
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "research-lock.json").exists() or (p / "crates" / "horizon-client").is_dir():
            return p
    raise FileNotFoundError("not inside the Horizon repository")


def default_binary():
    return Path(os.environ.get("HORIZON_RL_BINARY", repo_root() / "target/fast/horizon-client"))


def default_resources():
    return Path(os.environ.get("HORIZON_RL_RESOURCES", repo_root() / ".local/horizon-client/26.3"))


def scratch_env(root):
    """The environment of one game process: its own homes under `root`, no audio, no Discord."""
    env = dict(os.environ)
    for key in ("HORIZON_HOME", "HORIZON_LAUNCHER_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME"):
        d = Path(root) / key.lower()
        d.mkdir(parents=True, exist_ok=True)
        env[key] = str(d)
    env.update(HORIZON_NO_AUDIO="1", HORIZON_LAUNCHER_NO_SELF_UPDATE="1", HORIZON_NO_DISCORD="1",
               HORIZON_TELEMETRY_URL="off")
    return env


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_text(path, text, timeout, proc=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and text in path.read_text(errors="replace"):
            return
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"exited with {proc.returncode}; see {path}")
        time.sleep(0.2)
    raise TimeoutError(f"{text!r} not in {path} after {timeout}s")


class LocalServer:
    """`horizon-client --dedicated` on a flat, offline-mode world."""

    def __init__(self, root=None, port=None, binary=None, resources=None, properties=None):
        self.root = Path(root or tempfile.mkdtemp(prefix="horizon-rl-server-"))
        self.dir = self.root / "server"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.port = port or free_port()
        props = {"online-mode": "false", "server-ip": "127.0.0.1", "server-port": self.port,
                 "white-list": "false", "level-name": "world", "level-type": "minecraft:flat",
                 "generate-structures": "false", "spawn-protection": "0", "pvp": "true",
                 "difficulty": "easy", "gamemode": "survival", "view-distance": "4",
                 "simulation-distance": "4", "spawn-monsters": "false", "spawn-animals": "false",
                 "max-players": "64"}
        props.update(properties or {})
        (self.dir / "server.properties").write_text("".join(f"{k}={v}\n" for k, v in props.items()))
        (self.dir / "eula.txt").write_text("eula=true\n")
        self.log = self.dir / "out.log"
        cmd = [str(binary or default_binary()), "--resources", str(resources or default_resources()),
               "--dedicated", "--nogui"]
        self.proc = subprocess.Popen(cmd, cwd=self.dir, env=scratch_env(self.root / "home"),
                                     stdin=subprocess.PIPE, stdout=self.log.open("w"),
                                     stderr=subprocess.STDOUT, text=True)
        wait_for_text(self.log, "Done (", 300, self.proc)

    @property
    def address(self):
        return f"127.0.0.1:{self.port}"

    def command(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def wait_joined(self, name, timeout=120):
        wait_for_text(self.log, f"{name} joined the game", timeout, self.proc)

    def stop(self):
        if self.proc.poll() is None:
            try:
                self.command("stop")
                self.proc.wait(30)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill()


class ClientProcess:
    """One `horizon-client --headless` joined to `server` as `name`, with an RL socket unless
    `rl=False`."""

    def __init__(self, server_address, name, root, rl=True, render_distance=2, extra=(),
                 binary=None, resources=None):
        self.name = name
        self.root = Path(root) / name
        self.game_dir = self.root / "game"
        self.game_dir.mkdir(parents=True, exist_ok=True)
        self.socket = self.root / "rl.sock" if rl else None
        shm_dir = Path("/dev/shm") if Path("/dev/shm").is_dir() else self.root
        self.shm = shm_dir / f"horizon-rl-{os.getpid()}-{name}" if rl else None
        cmd = [str(binary or default_binary()), "--resources", str(resources or default_resources()),
               "--headless", "--game-dir", str(self.game_dir), "--server", server_address,
               "--name", name, "--render-distance", str(render_distance), "--no-pause", *extra]
        if rl:
            cmd += ["--rl-socket", str(self.socket), "--rl-shm", str(self.shm)]
        self.log = self.root / "out.log"
        self.proc = subprocess.Popen(cmd, cwd=self.game_dir, env=scratch_env(self.root / "home"),
                                     stdout=self.log.open("w"), stderr=subprocess.STDOUT)

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.shm is not None:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass


def remove(root):
    shutil.rmtree(root, ignore_errors=True)
