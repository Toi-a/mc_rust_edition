"""One game client, driven over its RL socket (docs/rl/design.md#transport)."""
import json
import math
import mmap
import os
import socket
import time

import numpy as np

from . import proto


class Disconnected(ConnectionError):
    """The game closed the session or died."""


class HorizonClient:
    """A session with one `horizon-client --rl-socket PATH`.

    `step` sends one action and returns `(state, info)` for the observation after it: `state` is
    a copy of the state vector (float32, the fair-play view), `info` a dict of reward signals.
    The layout is the game's own (`self.state_fields`: name -> slice).
    """

    def __init__(self, socket_path, frame_skip=1, connect_timeout=60.0, step_timeout=10.0):
        self.socket_path = str(socket_path)
        self.frame_skip = frame_skip
        self.step_timeout = step_timeout
        self.seq = 0
        self.sock = self._connect(connect_timeout)
        self._buf = b""
        self.sock.sendall(proto.hello(frame_skip))
        kind, payload = self._read(timeout=connect_timeout)
        if kind == proto.ERROR:
            raise Disconnected(payload.decode(errors="replace"))
        if kind != proto.WELCOME:
            raise Disconnected(f"expected WELCOME, got {kind:#x}")
        self.welcome = json.loads(payload)
        self.player = self.welcome["player"]
        self.items = self.welcome["items"]
        self.state_fields = {n: slice(o, o + l) for n, o, l in self.welcome["state"]}
        self.info_fields = {n: slice(o, o + l) for n, o, l in self.welcome["info"]}
        state_len = sum(l for _, _, l in self.welcome["state"])
        info_len = sum(l for _, _, l in self.welcome["info"])
        fd = os.open(self.welcome["shm"], os.O_RDONLY)
        try:
            self._mm = mmap.mmap(fd, self.welcome["shm_size"], prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        self._state = np.frombuffer(self._mm, np.float32, state_len, self.welcome["state_off"])
        self._info = np.frombuffer(self._mm, np.float32, info_len, self.welcome["info_off"])

    def _connect(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.connect(self.socket_path)
                return s
            except (FileNotFoundError, ConnectionRefusedError):
                s.close()
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)

    def _read(self, timeout):
        self.sock.settimeout(timeout)
        while True:
            if len(self._buf) >= 8:
                kind, length = proto._FRAME.unpack_from(self._buf)
                if len(self._buf) >= 8 + length:
                    payload = self._buf[8:8 + length]
                    self._buf = self._buf[8 + length:]
                    return kind, payload
            try:
                chunk = self.sock.recv(1 << 16)
            except socket.timeout as e:
                raise TimeoutError(f"no reply from {self.socket_path} in {timeout}s") from e
            except OSError as e:
                raise Disconnected(str(e)) from e
            if not chunk:
                raise Disconnected("the game closed the connection")
            self._buf += chunk

    def _ready(self, seq, timeout):
        while True:
            kind, payload = self._read(timeout)
            if kind == proto.ERROR:
                raise Disconnected(payload.decode(errors="replace"))
            if kind == proto.READY:
                got, tick = proto.read_ready(payload)
                if got == seq:
                    return tick

    def observation(self):
        """The current state vector (a copy) and info dict."""
        state = self._state.copy()
        info = {n: (float(self._info[s][0]) if s.stop - s.start == 1 else self._info[s].copy())
                for n, s in self.info_fields.items()}
        return state, info

    def step(self, buttons=0, hotbar=0, yaw=math.nan, pitch=math.nan, frame_skip=0):
        """Holds `buttons` (proto bits; attack is a click) from the next tick, faces `yaw`/`pitch`
        (absolute degrees, NaN keeps), and returns the observation after `frame_skip` ticks."""
        self.seq += 1
        self.sock.sendall(proto.step(self.seq, buttons, hotbar, yaw, pitch, frame_skip))
        self._ready(self.seq, self.step_timeout)
        return self.observation()

    def reset(self, timeout=120.0):
        """Waits until the player is alive in the world (after a death, the respawn), then
        returns its observation."""
        self.sock.sendall(proto.frame(proto.RESET))
        self._ready(0, timeout)
        return self.observation()

    def field(self, state, name):
        return state[self.state_fields[name]]

    def close(self):
        try:
            self.sock.sendall(proto.frame(proto.CLOSE))
        except OSError:
            pass
        self.sock.close()
        try:
            self._state = self._info = None
            self._mm.close()
        except (BufferError, AttributeError):
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
