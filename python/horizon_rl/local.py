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


# What a client that draws frames for images starts with (its game folder's options.txt, when it
# has none): no clouds, minimal particles, no smooth lighting, biome blend, mipmaps, shadows or
# fade-in. These cost drawing time and add nothing a policy needs; the view is otherwise vanilla's.
RENDER_OPTIONS = {
    "renderClouds": '"false"',
    "particles": "2",
    "ao": "false",
    "biomeBlendRadius": "0",
    "mipmapLevels": "0",
    "entityShadows": "false",
    "chunkSectionFadeInTime": "0.0",
    "enableVsync": "false",
    "weatherRadius": "3",
}


class ClientProcess:
    """One `horizon-client --headless` joined to `server` as `name`, with an RL socket unless
    `rl=False`. `peers`, `(name, target)` pairs, are more RL clients in the same process
    (`--rl-client`, process model B: they share its resources and GPU device), each with its own
    folder, socket and shared memory beside the first one's."""

    def __init__(self, server_address, name, root, rl=True, render_distance=2, extra=(),
                 binary=None, resources=None, env=None, render_options=None, peers=()):
        self.name = name
        self.root = Path(root) / name
        self.game_dir = self.root / "game"
        self.game_dir.mkdir(parents=True, exist_ok=True)
        options = self.game_dir / "options.txt"
        if render_options and not options.exists():
            options.write_text("".join(f"{k}:{v}\n" for k, v in render_options.items()))
        shm_dir = Path("/dev/shm") if Path("/dev/shm").is_dir() else self.root
        # Each client's socket and shared memory, by name: this one's first.
        self.sockets, self.shms = {}, {}
        if rl:
            self.sockets[name] = self.root / "rl.sock"
            self.shms[name] = shm_dir / f"horizon-rl-{os.getpid()}-{name}"
        cmd = [str(binary or default_binary()), "--resources", str(resources or default_resources()),
               "--headless", "--game-dir", str(self.game_dir), "--server", server_address,
               "--name", name, "--render-distance", str(render_distance), "--no-pause", *extra]
        if rl:
            cmd += ["--rl-socket", str(self.sockets[name]), "--rl-shm", str(self.shms[name])]
        for peer, target in peers:
            folder = Path(root) / peer
            folder.mkdir(parents=True, exist_ok=True)
            self.sockets[peer] = folder / "rl.sock"
            self.shms[peer] = shm_dir / f"horizon-rl-{os.getpid()}-{peer}"
            spec = f"name={peer},socket={self.sockets[peer]},shm={self.shms[peer]},dir={folder}"
            if any("," in str(v) for v in (peer, folder, self.shms[peer])):
                raise ValueError(f"--rl-client {spec}: no commas in names or paths")
            if target:
                spec += f",target={target}"
            cmd += ["--rl-client", spec]
        self.socket = self.sockets.get(name)
        self.shm = self.shms.get(name)
        self.log = self.root / "out.log"
        for path in self.sockets.values():
            path.unlink(missing_ok=True)  # a restart's: wait for the new one
        full_env = scratch_env(self.root / "home")
        full_env.update(env or {})
        with self.log.open("a") as log:
            self.proc = subprocess.Popen(cmd, cwd=self.game_dir, env=full_env, stdout=log,
                                         stderr=subprocess.STDOUT)

    def stop(self, timeout=15):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for shm in self.shms.values():
            try:
                shm.unlink()
            except FileNotFoundError:
                pass


def remove(root):
    shutil.rmtree(root, ignore_errors=True)
