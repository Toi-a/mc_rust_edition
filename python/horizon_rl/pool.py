"""The opponent pool: frozen versions of the learner that its opponents play
(docs/rl/design.md#the-opponent-pool).

The pool file is TOML, read once when a run starts:

    snapshot_every = 100_000   # learner steps between snapshots of the learner into the pool
    max_size = 50              # above it, a random run snapshot is dropped (never a listed version)
    new_weight = 1.0           # weight of a snapshot taken during the run
    newer_count_more = false   # true: each snapshot weighs recency_factor times the one before it
    recency_factor = 1.1

    [[version]]                # versions you chose yourself, each with its weight
    path = "runs/sword_a/snapshots/s001200000.zip"
    weight = 2.0

Relative paths are read from the current directory, the way the trainer is started. With no
`[[version]]` the pool starts with the learner's initial model, added as the run's first snapshot
(`Pool.add(path, 0)`) by whoever saved it.

This module holds the pool's bookkeeping only (no models, no game): which versions are in, their
weights, the draw, the snapshots and drops, and the learner's results against each version.
"""
import csv
import dataclasses
import math
import random
import tomllib
from pathlib import Path

RESULTS = ("win", "loss", "draw", "timeout", "abort")
# The stats columns, one per result.
COLUMNS = {"win": "wins", "loss": "losses", "draw": "draws", "timeout": "timeouts", "abort": "aborts"}


@dataclasses.dataclass(frozen=True)
class PoolConfig:
    snapshot_every: int = 100_000
    max_size: int = 50
    new_weight: float = 1.0
    newer_count_more: bool = False
    recency_factor: float = 1.1
    # (path, weight) of the versions listed in the file.
    versions: tuple = ()

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = tomllib.load(f)
        listed = data.pop("version", [])
        known = {f.name for f in dataclasses.fields(cls)} - {"versions"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"{path}: unknown keys {unknown} (known: {sorted(known)} and [[version]])")
        versions = []
        for i, v in enumerate(listed):
            extra = sorted(set(v) - {"path", "weight"})
            if "path" not in v or extra:
                raise ValueError(f"{path}: [[version]] {i + 1} needs a path (and a weight), not {extra}")
            versions.append((str(v["path"]), float(v.get("weight", 1.0))))
        return cls(versions=tuple(versions), **data)

    def __post_init__(self):
        if not (isinstance(self.snapshot_every, int) and self.snapshot_every > 0):
            raise ValueError(f"snapshot_every must be a positive number of steps, not {self.snapshot_every!r}")
        if not (isinstance(self.max_size, int) and self.max_size > 0):
            raise ValueError(f"max_size must be a positive number, not {self.max_size!r}")
        if not self.new_weight > 0:
            raise ValueError(f"new_weight must be above 0, not {self.new_weight}")
        if not self.recency_factor > 0:
            raise ValueError(f"recency_factor must be above 0, not {self.recency_factor}")
        if len(self.versions) >= self.max_size:
            raise ValueError(f"{len(self.versions)} versions are listed: they must be fewer than "
                             f"max_size ({self.max_size}), to leave room for the run's snapshots")
        paths = [p for p, _ in self.versions]
        doubled = sorted({p for p in paths if paths.count(p) > 1})
        if doubled:
            raise ValueError(f"versions listed twice: {doubled} (give one a higher weight instead)")
        for p, w in self.versions:
            if not (w > 0 and math.isfinite(w)):
                raise ValueError(f"version {p}: the weight must be above 0, not {w}")


@dataclasses.dataclass
class Version:
    """One frozen version in the pool. `id` never changes and is never reused, so an opponent in
    play keeps naming its version after a drop."""
    id: int
    path: str
    # Listed in the pool file (never dropped), or a snapshot of this run.
    listed: bool
    # Listed: the file's weight. A snapshot: the learner steps it was taken at.
    weight: float = 0.0
    step: int = 0
    # A snapshot's number among this run's snapshots (0 = the first), for newer_count_more.
    seq: int = 0

    @property
    def name(self):
        return Path(self.path).stem if self.listed else f"s{self.step:09d}"


class Pool:
    """The versions in play and the draw among them. `rng` is a `random.Random` (seed it for a
    reproducible run)."""

    def __init__(self, config, rng=None, check_files=True):
        self.config = config
        self.rng = rng or random.Random()
        self.versions = []
        self._next_id = 0
        self._snapshots = 0
        self.last_snapshot_step = None
        for path, weight in config.versions:
            if check_files and not Path(path).is_file():
                raise FileNotFoundError(f"pool version {path} doesn't exist (from {Path.cwd()})")
            self.versions.append(Version(self._new_id(), path, listed=True, weight=weight))

    def _new_id(self):
        self._next_id += 1
        return self._next_id - 1

    def __len__(self):
        return len(self.versions)

    def __iter__(self):
        return iter(self.versions)

    def get(self, id):
        return next((v for v in self.versions if v.id == id), None)

    @property
    def needs_initial(self):
        """Nothing listed: the run adds its initial model as its first snapshot."""
        return not self.versions

    def weight(self, v):
        if v.listed:
            return v.weight
        if not self.config.newer_count_more:
            return self.config.new_weight
        # The newest snapshot weighs new_weight; each older one 1/recency_factor of the next,
        # counted over every snapshot the run took (a drop changes no one else's weight).
        return self.config.new_weight * self.config.recency_factor ** (v.seq - (self._snapshots - 1))

    def probabilities(self):
        """{version id: its chance to be drawn}."""
        weights = [self.weight(v) for v in self.versions]
        total = sum(weights)
        return {v.id: w / total for v, w in zip(self.versions, weights)}

    def draw(self):
        """One opponent's version, in proportion to the weights. Each pair calls it on its own at
        every round end."""
        if not self.versions:
            raise RuntimeError("the pool is empty: add the initial model first")
        weights = [self.weight(v) for v in self.versions]
        return self.rng.choices(self.versions, weights=weights)[0]

    def due(self, step):
        """A snapshot is due at learner step `step` (counted over the whole run)."""
        if self.last_snapshot_step is None:
            return step >= self.config.snapshot_every
        return step - self.last_snapshot_step >= self.config.snapshot_every

    @staticmethod
    def snapshot_path(run_dir, step):
        """Where the snapshot at `step` is saved: `<run_dir>/snapshots/s001200000.zip`."""
        return Path(run_dir) / "snapshots" / f"s{step:09d}.zip"

    def add(self, path, step):
        """A snapshot of the learner at `step`, saved at `path`, joins the pool. Returns the
        version dropped to stay within max_size (a random run snapshot other than this one, never
        a listed version), or None. Its file stays on disk."""
        v = Version(self._new_id(), str(path), listed=False, step=step, seq=self._snapshots)
        self._snapshots += 1
        self.versions.append(v)
        self.last_snapshot_step = step
        if len(self.versions) <= self.config.max_size:
            return None
        older = [x for x in self.versions if not x.listed and x is not v]
        dropped = self.rng.choice(older)
        self.versions.remove(dropped)
        return dropped


def check_spaces(name, observation_space, action_space, run_observation_space, run_action_space):
    """Refuses a version whose spaces aren't the run's: its policy couldn't read the run's
    observations, or would send actions of another shape."""
    for what, theirs, ours in (("observation", observation_space, run_observation_space),
                               ("action", action_space, run_action_space)):
        if theirs != ours:
            raise ValueError(f"pool version {name}: its {what} space {theirs} isn't the run's {ours}")


class Stats:
    """The learner's results against each version: rounds, wins, losses, draws, timeouts and
    aborts, to pick the next run's versions and weights by."""

    FIELDS = ("version", "path", "rounds") + tuple(COLUMNS.values())

    def __init__(self):
        self.rows = {}  # version id -> {field: value}

    def record(self, version, result):
        """The learner's `result` ("win", "loss", ...) in a round against `version`."""
        if result not in RESULTS:
            raise ValueError(f"result {result!r}, not one of {RESULTS}")
        row = self.rows.setdefault(version.id, dict(
            {"version": version.name, "path": version.path, "rounds": 0}, **dict.fromkeys(COLUMNS.values(), 0)))
        row["rounds"] += 1
        row[COLUMNS[result]] += 1

    def write_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader()
            for row in self.rows.values():
                w.writerow(row)
        tmp.replace(path)
