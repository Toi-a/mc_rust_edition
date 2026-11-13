"""N headless clients from one config, kept alive by a supervisor
(docs/rl/design.md#fault-isolation-and-reconnects). Each runs in its own process, or with
`clients_per_process` K > 1, K slots share one (process model B, `--rl-client`: one copy of the
resources and one GPU device for K clients); a fault in any of them restarts the K together.

- Starts are staggered (`stagger` seconds apart, restarts included), so a server's connection
  throttle doesn't refuse them.
- A client that loses its server joins again by itself (the game's own backoff). The step that saw
  it is `truncated` with `info["reset_reason"] = "disconnected: <reason>"`.
- A process that exits, closes its socket, stops its heartbeat for `hang_timeout` seconds, or
  doesn't answer a step in `step_timeout` seconds is killed and started again after a backoff
  (1, 2, 4 ... 30 s). The step that saw it is `truncated` with `reset_reason` `crash`, `hang`
  or `timeout`. Nothing that happens to one slot stops the others.
- Episodes: `died` ends one (`terminated`). An ended slot is reset automatically: it is left out
  of the following steps (`active` False) until it is back alive, then its first observation comes
  with `info["episode_start"] = True`. A fault while it was between episodes (no step to truncate)
  is named in that first info's `reset_reason`.
"""
import dataclasses
import json
import math
import os
import selectors
import socket
import tempfile
import time
from pathlib import Path

import numpy as np

from . import proto
from .client import Framer, Session
from .local import RENDER_OPTIONS, ClientProcess, remove

# "state": the fair-play state vector; "image": grayscale frames of the view; "both". Every mode
# also returns the state vector (it costs nothing); "image" and "both" draw frames.
OBS_MODES = ("state", "image", "both")

# Backoff of a slot's restarts, in seconds; a slot healthy for RESTART_STABLE starts over.
RESTART_DELAYS = (1, 2, 4, 8, 16, 30)
RESTART_STABLE = 60.0


@dataclasses.dataclass
class FleetConfig:
    """One run's clients. `names` and `target` are format strings of the slot index `i`;
    `pairs` makes slots 2k and 2k+1 each other's target (it overrides `target`)."""
    server: str
    n: int = 2
    names: str = "Agent_{i}"
    target: str = ""
    pairs: bool = False
    obs: str = "state"
    # Images: (width, height), the width a multiple of 4; `hud` draws the HUD in them.
    image_size: tuple = (64, 64)
    hud: bool = False
    # The server shows its players each other's health (PvP servers do, above the name): the
    # state then has the opponent's health and absorption (`--rl-target-health`).
    target_health: bool = False
    frame_skip: int = 1
    render_distance: int = 2
    # Worker threads per client (chunk streaming jobs). Headless needs few.
    workers: int = 1
    # Clients per process (model B): slots i*K .. i*K+K-1 share one, and restart together.
    clients_per_process: int = 1
    stagger: float = 0.25
    hang_timeout: float = 5.0
    step_timeout: float = 10.0
    start_timeout: float = 60.0
    root: str = ""
    binary: str = ""
    resources: str = ""
    extra_args: tuple = ()

    @classmethod
    def load(cls, path, **overrides):
        """A config file (TOML, or JSON for `.json`), with keyword overrides on top."""
        path = Path(path)
        if path.suffix == ".json":
            data = json.loads(path.read_text())
        else:
            import tomllib
            data = tomllib.loads(path.read_text())
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown fleet settings: {', '.join(sorted(unknown))}")
        for key in ("extra_args", "image_size"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)

    @property
    def images(self):
        return self.obs in ("image", "both")

    def name(self, i):
        return self.names.format(i=i)

    def target_of(self, i):
        if self.pairs:
            partner = i ^ 1
            return self.name(partner) if partner < self.n else None
        return self.target.format(i=i) or None


class Slot:
    """One instance: its process, its socket and its shared memory, and where it stands."""
    # backoff -> starting -> hello -> idle -> resetting -> ready <-> stepping; any -> backoff
    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.state = "backoff"
        self.proc = None
        self.sock = None
        self.framer = None
        self.session = None
        self.seq = 0
        self.until = 0.0  # backoff end, or the deadline of what it waits for
        self.restarts = 0
        self.attempts = 0
        self.healthy_since = None
        self.beat = (-1, 0.0)
        self.notes = []
        self.reason = None  # why the episode ended, for the step that reports it
        self.start_reason = None  # a fault between episodes, for the next one's first info
        self.last = None  # the last (state, info)
        self.last_frame = 0  # and its frame, with images
        self.episode_start = False
        self.need_reset = True
        self.group = [self]  # the slots sharing its process, itself included


class Fleet:
    """The clients of one `FleetConfig`. `start()`, then `reset()`, then `step(actions)`."""

    def __init__(self, config):
        if config.obs not in OBS_MODES:
            raise ValueError(f"obs {config.obs!r}: one of {OBS_MODES}")
        w, h = config.image_size
        if config.images and (w % 4 or not 8 <= w <= 512 or not 8 <= h <= 512):
            raise ValueError(f"image_size {config.image_size}: sides 8..512, the width a multiple of 4")
        self.config = config
        self.root = Path(config.root or tempfile.mkdtemp(prefix="horizon-rl-fleet-"))
        self.root.mkdir(parents=True, exist_ok=True)
        if config.clients_per_process < 1:
            raise ValueError("clients_per_process: at least 1")
        self.slots = [Slot(i, config.name(i)) for i in range(config.n)]
        k = config.clients_per_process
        # The slots of one process: its first starts it, the others are its `--rl-client`s.
        self.groups = [self.slots[i:i + k] for i in range(0, config.n, k)]
        for group in self.groups:
            for slot in group:
                slot.group = group
        self.selector = selectors.DefaultSelector()
        self.next_start = time.monotonic()
        self.state_len = None
        self.state_fields = None
        self.info_fields = None
        self.items = None
        # The last frames, (n, h, w) uint8, with images (zeros for a slot without one).
        self.frames = (np.zeros((config.n, config.image_size[1], config.image_size[0]), np.uint8)
                       if config.images else None)

    # -- processes ---------------------------------------------------------------------------

    def _launch(self, group, now):
        c = self.config
        slot = group[0]
        extra = list(c.extra_args)
        target = c.target_of(slot.index)
        if target:
            extra += ["--rl-target", target]
        if c.target_health:
            extra.append("--rl-target-health")
        env = {"HORIZON_JOB_THREADS": str(c.workers), "HORIZON_WORKER_THREADS": str(c.workers)}
        render_options = None
        if c.images:
            extra += ["--rl-image", f"{c.image_size[0]}x{c.image_size[1]}"]
            if c.hud:
                extra.append("--rl-hud")
            # One frame per observation, read back at once: no thread to submit it on.
            env["HORIZON_SUBMIT_THREAD"] = "0"
            # Uploads in system memory, not the GPU's small BAR1 window, which every process
            # shares: N clients mapping blocks there run it out (third_party/wgpu-hal).
            env["HORIZON_NO_BAR_UPLOADS"] = os.environ.get("HORIZON_NO_BAR_UPLOADS", "1")
            # The terrain arena's first page: a 64x64 view at render distance 2 fills ~5 MB of
            # the default ~30 MB (it grows on demand) (gfx/arena.rs).
            env["HORIZON_SECTION_ARENA_VERTICES"] = os.environ.get("HORIZON_SECTION_ARENA_VERTICES",
                                                                   str(1 << 18))
            # The meshing workers keep 1 MB of cleared mesh buffers, not 16 (streaming/mesh_pool.rs).
            env["HORIZON_MESH_POOL_BYTES"] = os.environ.get("HORIZON_MESH_POOL_BYTES", str(1 << 20))
            # Without the HUD nothing of the GUI is drawn: its textures and the menus' slots stay
            # out of the atlas, 31 MiB of GPU memory instead of 96 (hud/slots.rs).
            if not c.hud:
                env["HORIZON_LEAN_ATLAS"] = os.environ.get("HORIZON_LEAN_ATLAS", "1")
            render_options = RENDER_OPTIONS
        peers = [(s.name, c.target_of(s.index)) for s in group[1:]]
        proc = ClientProcess(c.server, slot.name, self.root, render_distance=c.render_distance,
                             extra=extra, binary=c.binary or None, resources=c.resources or None,
                             env=env, render_options=render_options, peers=peers)
        for s in group:
            s.proc = proc
            s.state = "starting"
            s.until = now + c.start_timeout

    def _fail(self, slot, reason, now):
        """Kills the slot's process and plans its restart: with the other slots of that process,
        which end with the same reason, "(shard)" added."""
        if slot.healthy_since is not None and now - slot.healthy_since >= RESTART_STABLE:
            slot.attempts = 0
        if slot.proc is not None:
            slot.proc.stop(timeout=2)
        delay = RESTART_DELAYS[min(slot.attempts, len(RESTART_DELAYS) - 1)]
        slot.attempts += 1
        for s in slot.group:
            if s is not slot and s.state == "backoff":
                continue
            self._close_conn(s)
            s.proc = None
            s.attempts = max(s.attempts, slot.attempts)
            s.restarts += 1
            s.healthy_since = None
            s.state = "backoff"
            s.until = now + delay
            s.need_reset = True
            if s.reason is None:
                s.reason = reason if s is slot else f"{reason} (shard)"

    def _close_conn(self, slot):
        if slot.sock is not None:
            try:
                self.selector.unregister(slot.sock)
            except (KeyError, ValueError):
                pass
            slot.sock.close()
            slot.sock = None
        if slot.session is not None:
            slot.session.close()
            slot.session = None

    # -- the loop ----------------------------------------------------------------------------

    def _pump(self, timeout):
        """One round: start what is due, read every socket, watch heartbeats and deadlines."""
        now = time.monotonic()
        for group in self.groups:
            if (now >= self.next_start and all(s.state == "backoff" and now >= s.until for s in group)):
                self._launch(group, now)
                self.next_start = now + self.config.stagger
        for slot in self.slots:
            if slot.state == "starting":
                self._try_connect(slot, now)
        for key, _ in self.selector.select(timeout):
            # A failure earlier in this batch may have closed this socket: a shard's process
            # dying shows on all its sockets at once, and the first one read restarts them all.
            if key.data.sock is key.fileobj:
                self._read(key.data, time.monotonic())
        now = time.monotonic()
        for slot in self.slots:
            if slot.state in ("backoff",):
                continue
            if slot.proc is not None and slot.proc.proc.poll() is not None:
                self._fail(slot, f"crash (exit {slot.proc.proc.returncode})", now)
                continue
            if slot.session is not None:
                beat = slot.session.beat()
                if beat != slot.beat[0]:
                    slot.beat = (beat, now)
                elif now - slot.beat[1] > self.config.hang_timeout:
                    self._fail(slot, "hang", now)
                    continue
            if slot.state in ("starting", "hello", "stepping") and now > slot.until:
                self._fail(slot, "timeout" if slot.state == "stepping" else "start timeout", now)

    def _try_connect(self, slot, now):
        path = slot.proc.sockets[slot.name]
        if not path.exists():
            if now > slot.until:
                self._fail(slot, "start timeout", now)
            return
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            s.connect(str(path))
        except (FileNotFoundError, ConnectionRefusedError):
            s.close()
            return
        s.setblocking(False)
        slot.sock = s
        slot.framer = Framer()
        self.selector.register(s, selectors.EVENT_READ, slot)
        s.sendall(proto.hello(self.config.frame_skip))
        slot.state = "hello"

    def _read(self, slot, now):
        try:
            data = slot.sock.recv(1 << 16)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if not data:
            self._fail(slot, "crash (socket closed)", now)
            return
        slot.framer.push(data)
        while slot.sock is not None and (m := slot.framer.next()) is not None:
            self._handle(slot, *m, now)

    def _handle(self, slot, kind, payload, now):
        if kind == proto.WELCOME and slot.state == "hello":
            slot.session = Session(json.loads(payload))
            slot.beat = (slot.session.beat(), now)
            if self.state_fields is None:
                self.state_len = slot.session.state_len
                self.state_fields = slot.session.state_fields
                self.info_fields = slot.session.info_fields
                self.items = slot.session.items
            slot.state = "idle"
            slot.need_reset = True
        elif kind == proto.NOTE:
            note = json.loads(payload)
            slot.notes.append(note)
            if note.get("event") == "disconnected":
                slot.reason = "disconnected: " + note.get("reason", "")
        elif kind == proto.READY:
            seq, _tick = proto.read_ready(payload)
            if slot.state == "resetting" and seq == 0:
                self._observe(slot)
                slot.state = "ready"
                slot.need_reset = False
                slot.episode_start = True
                slot.start_reason, slot.reason = slot.reason, None
                if slot.healthy_since is None:
                    slot.healthy_since = now
            elif slot.state == "stepping" and seq == slot.seq:
                self._observe(slot)
                slot.state = "ready"
        elif kind == proto.ERROR:
            self._fail(slot, "refused: " + payload.decode(errors="replace"), now)

    def _observe(self, slot):
        slot.last = slot.session.observation()
        if self.frames is not None:
            frame = slot.session.frame()
            slot.last_frame = frame if frame is not None else 0

    def _send(self, slot, data, now):
        try:
            slot.sock.sendall(data)
            return True
        except OSError:
            self._fail(slot, "crash (socket closed)", now)
            return False

    def _begin_reset(self, slot, now):
        if self._send(slot, proto.frame(proto.RESET), now):
            slot.state = "resetting"

    # -- the API -----------------------------------------------------------------------------

    def start(self, timeout=None):
        """Starts every process (staggered) and waits until each has a session."""
        deadline = time.monotonic() + (timeout or self.config.start_timeout + self.config.n * self.config.stagger)
        while any(s.state in ("backoff", "starting", "hello") for s in self.slots):
            if time.monotonic() > deadline:
                missing = [s.name for s in self.slots if s.state in ("backoff", "starting", "hello")]
                raise TimeoutError(f"not started: {', '.join(missing)}")
            self._pump(0.05)
        return self

    def reset(self, timeout=120.0):
        """Resets every slot and waits until all are alive; returns states (n, L) and infos."""
        now = time.monotonic()
        for slot in self.slots:
            if slot.state in ("idle", "ready"):
                self._begin_reset(slot, now)
        deadline = now + timeout
        while any(s.state != "ready" for s in self.slots):
            if time.monotonic() > deadline:
                raise TimeoutError("not all slots came back: " + ", ".join(
                    f"{s.name}={s.state}" for s in self.slots if s.state != "ready"))
            self._pump(0.05)
            now = time.monotonic()
            for slot in self.slots:
                if slot.state == "idle":
                    self._begin_reset(slot, now)
        states = np.stack([s.last[0] for s in self.slots])
        infos = [dict(s.last[1], episode_start=True) for s in self.slots]
        if self.frames is not None:
            for i, s in enumerate(self.slots):
                self.frames[i] = s.last_frame
        for s in self.slots:
            s.episode_start = False
        return states, infos

    def step(self, actions):
        """`actions[i]` is `(buttons, hotbar, yaw, pitch)` for slot `i` (ignored when it isn't
        active). Returns `states (n, L), infos, terminated, truncated, active`; with images,
        `self.frames` (n, h, w) holds the frames of the same observations."""
        now = time.monotonic()
        budget = self.config.step_timeout + self.config.frame_skip * 0.05
        for slot in self.slots:
            if slot.state == "idle":  # (re)started: its episode begins once it is alive
                self._begin_reset(slot, now)
        stepped = []
        for slot, action in zip(self.slots, actions):
            if slot.state != "ready" or slot.need_reset or slot.episode_start:
                continue
            slot.seq += 1
            buttons, hotbar, yaw, pitch = action
            if self._send(slot, proto.step(slot.seq, buttons, hotbar, yaw, pitch), now):
                slot.state = "stepping"
                slot.until = now + budget
                stepped.append(slot)
        if stepped:
            while any(s.state == "stepping" for s in stepped):
                self._pump(budget)
        else:
            # Every slot is between episodes: wait about a tick for one to come back.
            self._pump(0.05)
        n = len(self.slots)
        states = np.zeros((n, self.state_len), np.float32)
        infos = [{} for _ in range(n)]
        terminated = np.zeros(n, bool)
        truncated = np.zeros(n, bool)
        active = np.zeros(n, bool)
        now = time.monotonic()
        for i, slot in enumerate(self.slots):
            if slot.episode_start and slot.state == "ready":
                # Back from a reset: this is the new episode's first observation.
                states[i], info = slot.last
                infos[i] = dict(info, episode_start=True)
                if slot.start_reason:
                    infos[i]["reset_reason"] = slot.start_reason
                    slot.start_reason = None
                active[i] = True
                slot.episode_start = False
                continue
            if slot not in stepped:
                infos[i] = {"state": slot.state, "restarts": slot.restarts}
                continue
            active[i] = True
            if slot.state != "ready":
                # It failed during the step: the last observation stands, and it ends here.
                states[i] = slot.last[0] if slot.last else 0
                truncated[i] = True
                infos[i] = {"reset_reason": slot.reason or "crash", "restarts": slot.restarts}
                slot.reason = None
                continue
            states[i], info = slot.last
            infos[i] = info
            if info.get("died", 0) > 0:
                terminated[i] = True
            elif info.get("disconnected", 0) > 0 or info.get("connected", 1) == 0:
                truncated[i] = True
                info["reset_reason"] = slot.reason or "disconnected"
                slot.reason = None
            if terminated[i] or truncated[i]:
                self._begin_reset(slot, now)
        if self.frames is not None:
            for i, slot in enumerate(self.slots):
                self.frames[i] = slot.last_frame if active[i] else 0
        return states, infos, terminated, truncated, active

    def stats(self):
        return [{"name": s.name, "state": s.state, "restarts": s.restarts,
                 "pid": s.proc.proc.pid if s.proc else None, "notes": len(s.notes)}
                for s in self.slots]

    def kill(self, i):
        """Kills slot `i`'s process (a test of the supervisor)."""
        if self.slots[i].proc is not None:
            self.slots[i].proc.proc.kill()

    def close(self, keep_files=False):
        for slot in self.slots:
            self._close_conn(slot)
            if slot.proc is not None:
                slot.proc.stop()
                slot.proc = None
        self.selector.close()
        if not keep_files and not self.config.root:
            remove(self.root)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def nan_action():
    """Holds nothing and keeps the view."""
    return (0, 0, math.nan, math.nan)
