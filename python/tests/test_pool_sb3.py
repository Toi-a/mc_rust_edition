"""Training against the opponent pool (OpponentPoolVecEnv and its callbacks) over the fake fleet
of test_env (no game). Needs stable-baselines3."""
import csv
import dataclasses
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_env import FakeFleet, with_opponent  # noqa: E402

from horizon_rl.env import HorizonVecEnv  # noqa: E402
from horizon_rl.pool import Pool, PoolConfig  # noqa: E402

try:
    from horizon_rl.pool_sb3 import OpponentPoolVecEnv, PoolSnapshots, PoolStats
    from horizon_rl.sb3 import HorizonSB3VecEnv, ReleaseKeys
except ImportError:
    OpponentPoolVecEnv = None

BUTTONS = ("forward", "back", "jump", "attack")


class FakeModel:
    """A frozen version: it presses `forward` only if `presses`, and keeps the batches it saw."""

    def __init__(self, env, presses):
        self.observation_space, self.action_space = env.observation_space, env.action_space
        self.presses = presses
        self.batches = []

    def predict(self, obs, deterministic=True):
        self.batches.append((len(obs), deterministic))
        a = np.zeros((len(obs), *self.action_space.shape), np.int64)
        a[:, 0] = int(self.presses)
        return a, None


def make(n=4, script=(), signals=None, versions=(("a.zip", 1.0), ("b.zip", 1.0)), seed=0, **pool):
    fleet = with_opponent(FakeFleet(n, "state", script=script, signals=signals))
    fleet.config = dataclasses.replace(fleet.config, pairs=True)
    inner = HorizonSB3VecEnv(HorizonVecEnv(fleet=fleet, camera="aim", hotbar=False, flat=True, buttons=BUTTONS))
    models = {}

    def load(path):
        # "a" presses forward, the others don't: the sent actions say who played which seat.
        models[path] = FakeModel(inner, presses=Path(path).stem == "a")
        return models[path]

    p = Pool(PoolConfig(versions=versions, **pool), rng=random.Random(seed), check_files=False)
    return OpponentPoolVecEnv(inner, p, load=load), fleet, models


def sent_forward(fleet, step=-1):
    """Per agent: did the step's action press forward (bit 1)."""
    return [bits & 1 for bits, *_ in fleet.sent[step]]


@unittest.skipIf(OpponentPoolVecEnv is None, "stable-baselines3 isn't installed")
class PoolEnvTest(unittest.TestCase):
    def test_sb3_sees_the_learner_seats_only(self):
        env, fleet, models = make(n=4)
        self.assertEqual(env.num_envs, 2)
        obs = env.reset()
        self.assertEqual(obs.shape, (2, 8))
        self.assertEqual(sorted(models), ["a.zip", "b.zip"], "every listed version is loaded at the start")
        learner = np.zeros((2, 4), np.int64)
        learner[:, 3] = 1  # attack
        obs, rewards, dones, infos = env.step(learner)
        self.assertEqual((obs.shape, rewards.shape, dones.shape, len(infos)), ((2, 8), (2,), (2,), 2))
        bits = [b for b, *_ in fleet.sent[-1]]
        # Seats 0 and 2 are the learner's (attack = 128); 1 and 3 the opponents' (no attack).
        self.assertEqual([b & 128 for b in bits], [128, 0, 128, 0])
        for model in models.values():
            for size, deterministic in model.batches:
                self.assertFalse(deterministic, "opponents sample their actions")
        self.assertEqual(sum(size for m in models.values() for size, _ in m.batches), 2,
                         "one forward pass per version in play, over its pairs")

    def test_each_pair_draws_again_when_its_round_ends_and_stats_count_the_result(self):
        # Pair 0's round ends on step 2 (the learner wins); pair 1 plays on.
        script = [["play"] * 4, ["die", "play", "play", "play"]] + [["play"] * 4] * 40
        env, fleet, _ = make(n=4, script=script, signals={(3, 0): {"round_win": 1.0, "round_end": 1.0}})
        env.reset()
        env.step(np.zeros((2, 4), np.int64))
        first = list(env.playing)
        _, _, dones, infos = env.step(np.zeros((2, 4), np.int64))
        self.assertEqual(dones.tolist(), [True, False])
        self.assertEqual(infos[0]["opponent_version"], first[0].name)
        self.assertEqual(env.playing[1], first[1], "the other pair keeps its opponent")
        row = env.stats.rows[first[0].id]
        self.assertEqual((row["rounds"], row["wins"], row["losses"]), (1, 1, 0))
        # The new opponent plays the next steps: its seat presses forward exactly when it is "a".
        env.step(np.zeros((2, 4), np.int64))
        self.assertEqual(sent_forward(fleet)[1], int(env.playing[0].path == "a.zip"))
        self.assertEqual(sent_forward(fleet)[3], int(env.playing[1].path == "a.zip"))

    def test_draws_follow_the_weights(self):
        script = [["die", "play"]] * 2000
        env, _, _ = make(n=2, script=script, versions=(("a.zip", 3.0), ("b.zip", 1.0)))
        env.reset()
        drawn = []
        for _ in range(2000):
            env.step(np.zeros((1, 4), np.int64))
            drawn.append(env.playing[0].path)
        self.assertAlmostEqual(drawn.count("a.zip") / len(drawn), 0.75, delta=0.04)

    def test_a_version_with_other_spaces_is_refused_at_the_start(self):
        from gymnasium import spaces
        fleet = with_opponent(FakeFleet(2, "state"))
        fleet.config = dataclasses.replace(fleet.config, pairs=True)
        inner = HorizonSB3VecEnv(HorizonVecEnv(fleet=fleet, camera="aim", hotbar=False, flat=True, buttons=BUTTONS))
        other = FakeModel(inner, False)
        other.observation_space = spaces.Box(-1, 1, (3,))
        pool = Pool(PoolConfig(versions=(("old.zip", 1.0),)), check_files=False)
        with self.assertRaisesRegex(ValueError, "old.*observation"):
            OpponentPoolVecEnv(inner, pool, load=lambda path: other)

    def test_a_dropped_version_plays_its_round_out_then_is_unloaded(self):
        env, _, models = make(n=2, script=[["play", "play"]] * 3 + [["die", "play"]] + [["play", "play"]] * 3,
                              versions=(), max_size=1)
        model = SimpleSaver()
        env.add_snapshot(model, Path(tempfile.mkdtemp()) / "s1.zip", 0)
        env.reset()
        env.step(np.zeros((1, 4), np.int64))
        old = env.playing[0]
        env.add_snapshot(model, Path(tempfile.mkdtemp()) / "s2.zip", 100)
        self.assertNotIn(old, list(env.pool), "dropped above max_size")
        self.assertIn(old.id, env.models, "still loaded: its pair is playing it")
        for _ in range(3):
            env.step(np.zeros((1, 4), np.int64))
        self.assertNotEqual(env.playing[0], old)
        self.assertNotIn(old.id, env.models, "unloaded once its round ended")

    def test_release_lets_go_of_every_agents_keys(self):
        env, fleet, _ = make(n=4)
        env.reset()
        env.env_method("release", indices=[0])
        self.assertTrue(all(np.isnan(a[2]) and a[0] == 0 for a in fleet.sent[-1]), "all four seats")


class SimpleSaver:
    def save(self, path):
        Path(path).write_bytes(b"")


@unittest.skipIf(OpponentPoolVecEnv is None, "stable-baselines3 isn't installed")
class PoolTrainingTest(unittest.TestCase):
    def test_ppo_trains_against_its_snapshots(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecMonitor
        rounds = {(t, 0): {"round_end": 1.0, "round_loss": 1.0} for t in range(5, 200, 6)}
        script = [["die" if (t + 2) % 6 == 5 else "play", "play"] for t in range(200)]
        fleet = with_opponent(FakeFleet(2, "state", script=script, signals=rounds))
        fleet.config = dataclasses.replace(fleet.config, pairs=True)
        inner = HorizonSB3VecEnv(HorizonVecEnv(fleet=fleet, camera="aim", hotbar=False, flat=True, buttons=BUTTONS))
        with tempfile.TemporaryDirectory() as run:
            env = OpponentPoolVecEnv(inner, Pool(PoolConfig(snapshot_every=16)), opponent_device="cpu")
            model = PPO("MlpPolicy", VecMonitor(env), n_steps=8, batch_size=8, n_epochs=1, device="cpu")
            model.learn(48, callback=[ReleaseKeys(), PoolSnapshots(run, verbose=0), PoolStats(run)])
            snaps = sorted(p.name for p in (Path(run) / "snapshots").iterdir())
            self.assertEqual(snaps, ["s000000000.zip", "s000000016.zip", "s000000032.zip", "s000000048.zip"])
            self.assertEqual(len(env.pool), 4)
            self.assertEqual(len(env.models), 4, "each loaded from its file")
            with open(Path(run) / "pool_stats.csv") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(sum(int(r["rounds"]) for r in rows), env.rounds_done)
            self.assertGreater(env.rounds_done, 0)
            self.assertEqual(sum(int(r["losses"]) for r in rows), env.rounds_done)


if __name__ == "__main__":
    unittest.main()
