"""The wire format of crates/horizon-rl-proto, as Python sees it.

Message kinds, button bits and the fixed-size structs are mirrored here. The state and info layouts
are not: they come from the game's WELCOME reply, so they can't drift from the Rust side.
"""
import struct

VERSION = 2
SHM_MAGIC = b"HZRL"
HEADER_LEN = 64
# The game bumps a u64 here every frame (in or out of a world): the supervisor's hang check.
BEAT_OFF = 56

HELLO, STEP, RESET, CLOSE = 1, 2, 3, 4
# NOTE: JSON about the connection, sent at any time ({"event": "disconnected", "reason": ...}).
WELCOME, READY, NOTE, ERROR = 0x81, 0x82, 0x83, 0xFF

FORWARD = 1 << 0
BACK = 1 << 1
LEFT = 1 << 2
RIGHT = 1 << 3
JUMP = 1 << 4
SPRINT = 1 << 5
SNEAK = 1 << 6
ATTACK = 1 << 7
USE = 1 << 8
BUTTONS = {"forward": FORWARD, "back": BACK, "left": LEFT, "right": RIGHT, "jump": JUMP,
           "sprint": SPRINT, "sneak": SNEAK, "attack": ATTACK, "use": USE}

_FRAME = struct.Struct("<II")
_HELLO = struct.Struct("<II")
_STEP = struct.Struct("<QIIIff")
_READY = struct.Struct("<QQ")
_HEADER = struct.Struct("<4sIQQIIIIIII")


def frame(kind, payload=b""):
    return _FRAME.pack(kind, len(payload)) + payload


def hello(frame_skip):
    return frame(HELLO, _HELLO.pack(VERSION, frame_skip))


def step(seq, buttons, hotbar, yaw, pitch, frame_skip=0):
    """`yaw`/`pitch` are absolute degrees; NaN keeps the current angle. `hotbar` 0 keeps the slot."""
    return frame(STEP, _STEP.pack(seq, frame_skip, buttons, hotbar, yaw, pitch))


def read_ready(payload):
    return _READY.unpack(payload)


def read_header(buf):
    magic, version, seq, tick, state_off, state_len, info_off, info_len, frame_off, w, h = \
        _HEADER.unpack_from(buf, 0)
    return {"magic": magic, "version": version, "seq": seq, "tick": tick, "state_off": state_off,
            "state_len": state_len, "info_off": info_off, "info_len": info_len,
            "frame_off": frame_off, "frame_w": w, "frame_h": h}


def read_beat(buf):
    return struct.unpack_from("<Q", buf, BEAT_OFF)[0]


def buttons_of(action):
    """A dict like {"forward": True, "attack": 1} as button bits."""
    bits = 0
    for name, bit in BUTTONS.items():
        if action.get(name):
            bits |= bit
    return bits
