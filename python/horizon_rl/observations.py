"""The state observation, chosen per run: `ObservationView` builds a policy's input from the
game's state vector (docs/rl/README.md#observations).

    HorizonVecEnv(..., observation=("self_vel_local", "self_health", "target_distance"))
    HorizonVecEnv(..., observation="sword")          # a preset (PRESETS)

Every field of the state vector is a feature under its own name (`env.state_fields`), and there
are derived ones (`FEATURES` lists all):

- `self_vel_local`, `target_rel_vel_local`, `target_rel_pos_local`: the vector in the agent's own
  frame, (forward, right, up): forward is where the agent faces (its yaw; the pitch isn't used),
  right is its right hand, up is up. The game's vectors are along the world's x, y, z.
- `self_yaw_sincos`, `target_bearing_sincos`: an angle as (sin, cos), which has no jump at
  +-180 degrees (the bearing's also carries the pitch part, so it has 3 values).

With `normalize=True` (the default) each feature is divided by its usual range (`SCALES`), so
values are mostly within -1..1: health by 20, velocities by 0.5 blocks per tick, distances by 8
blocks, angles by 180 (yaw) or 90 (pitch) degrees, the hurt flash by its 10 ticks. Nothing is
clipped. Item ids and the world position have no scale: use them raw, or not at all.

`target_health` and `target_absorption` are only filled when the run says the server shows
players each other's health (`FleetConfig.target_health`, set for you when an observation asks
for them): see the fair-play rules in docs/rl/design.md.
"""
import numpy as np

# What each state field is divided by with `normalize=True` (1 where it isn't listed).
SCALES = {
    "self_vel": 0.5, "self_health": 20.0, "self_absorption": 20.0, "self_food": 20.0,
    "self_armor": 20.0, "self_selected": 8.0, "self_hotbar_count": 64.0, "self_offhand_count": 64.0,
    "self_pitch": 90.0, "self_hurt_time": 10.0,
    "target_rel_pos": 8.0, "target_rel_vel": 0.5, "target_distance": 8.0, "target_pitch": 90.0,
    "target_hurt_time": 10.0, "target_health": 20.0, "target_absorption": 20.0,
}
# Yaws: folded into -180..180 (the game's own yaw keeps counting past it), then divided by 180.
YAWS = ("self_yaw", "target_yaw")
# Features that need the server-shown health (`--rl-target-health`).
SHOWN_HEALTH = ("target_health", "target_absorption")


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _local(vectors, yaw):
    """World (x, y, z) vectors, (n, 3), as (forward, right, up) of someone facing `yaw` degrees
    (Minecraft's: 0 faces +z, 90 faces -x)."""
    sin, cos = np.sin(np.radians(yaw)), np.cos(np.radians(yaw))
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    return np.stack([-x * sin + z * cos, -x * cos - z * sin, y], axis=1)


def _derived(name, scale):
    """name -> (size, the state fields it reads, function of (states, fields, normalize))."""
    def local(source):
        def build(s, f, normalize):
            out = _local(s[:, f[source]], s[:, f["self_yaw"].start])
            return out / scale[source] if normalize else out
        return 3, (source, "self_yaw"), build

    def yaw_sincos(s, f, normalize):
        yaw = np.radians(s[:, f["self_yaw"].start])
        return np.stack([np.sin(yaw), np.cos(yaw)], axis=1)

    def bearing_sincos(s, f, normalize):
        bearing = s[:, f["target_bearing"]]
        yaw = np.radians(bearing[:, 0])
        present = s[:, f["target_present"].start] > 0.5
        # (0, 0, 0) without an opponent, like every other target feature.
        pitch = bearing[:, 1] / 90.0 if normalize else bearing[:, 1]
        return np.stack([np.sin(yaw), np.cos(yaw), pitch], axis=1) * present[:, None]

    def bearing(s, f, normalize):
        out = s[:, f["target_bearing"]]
        return out / np.array([180.0, 90.0], np.float32) if normalize else out

    return {
        "self_vel_local": local("self_vel"),
        "target_rel_vel_local": local("target_rel_vel"),
        "target_rel_pos_local": local("target_rel_pos"),
        "self_yaw_sincos": (2, ("self_yaw",), yaw_sincos),
        "target_bearing_sincos": (3, ("target_bearing", "target_present"), bearing_sincos),
        "target_bearing": (2, ("target_bearing",), bearing),
    }[name]


DERIVED = ("self_vel_local", "target_rel_vel_local", "target_rel_pos_local", "self_yaw_sincos",
           "target_bearing_sincos", "target_bearing")

# Named choices of features. "sword": the first training stage (a sword duel with aim assist,
# on a server that shows health above the players).
PRESETS = {
    "sword": ("self_vel_local", "self_yaw", "self_pitch", "self_health", "self_attack_cooldown",
              "self_on_ground", "self_hurt_time", "self_alive",
              "target_present", "target_rel_vel_local", "target_distance", "target_health"),
}


def features(state_fields):
    """Every feature name there is, given the game's state layout."""
    return sorted(set(state_fields) | set(DERIVED))


class ObservationView:
    """`names`: features in the order they are laid out, or a preset's name. `bind` it to the
    game's layout, then `build(states)` gives the (n, size) float32 observations and `fields`
    says where each feature is."""

    def __init__(self, names, normalize=True, scales=None):
        if isinstance(names, str):
            if names not in PRESETS:
                raise ValueError(f"observation {names!r}: a list of features, or one of {sorted(PRESETS)}")
            names = PRESETS[names]
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("observation: at least one feature, each once")
        self.normalize = bool(normalize)
        self.scales = dict(SCALES, **(scales or {}))
        self.needs_target_health = any(n in SHOWN_HEALTH for n in self.names)
        self.fields = None
        self.size = 0
        self._parts = []

    def bind(self, state_fields):
        self.fields, self._parts, at = {}, [], 0
        for name in self.names:
            if name in DERIVED:
                size, needs, build = _derived(name, self.scales)
            elif name in state_fields:
                size, needs, build = state_fields[name].stop - state_fields[name].start, (name,), None
            else:
                raise ValueError(f"observation feature {name!r}: one of {features(state_fields)}")
            missing = [n for n in needs if n not in state_fields]
            if missing:
                raise ValueError(f"observation feature {name!r} needs {missing} from the game: rebuild it "
                                 "(cargo build --profile fast -p horizon-client --bins)")
            self.fields[name] = slice(at, at + size)
            self._parts.append((name, build))
            at += size
        self.size = at
        self._state_fields = state_fields
        return self

    def build(self, states):
        f = self._state_fields
        out = np.empty((len(states), self.size), np.float32)
        for name, build in self._parts:
            if build is not None:
                value = build(states, f, self.normalize)
            else:
                value = states[:, f[name]]
                if self.normalize:
                    value = _wrap(value) / 180.0 if name in YAWS else value / self.scales.get(name, 1.0)
            out[:, self.fields[name]] = value
        return out
