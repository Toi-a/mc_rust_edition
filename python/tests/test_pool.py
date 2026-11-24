"""The opponent pool's bookkeeping: the pool file, weights, the draw, snapshots and drops, stats
(no game, no models)."""
import collections
import csv
import random
import sys
import tempfile
import unittest
from pathlib import Path

from gymnasium import spaces

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl.pool import Pool, PoolConfig, Stats, check_spaces  # noqa: E402


def pool_file(text):
    d = tempfile.TemporaryDirectory()
    path = Path(d.name) / "pool.toml"
    path.write_text(text)
    return d, path


class PoolFile(unittest.TestCase):
    def test_reads_the_settings_and_the_listed_versions(self):
        d, path = pool_file("""
            snapshot_every = 50_000
            max_size = 4
            new_weight = 0.5
            newer_count_more = true
            recency_factor = 2.0
            [[version]]
            path = "runs/a/snapshots/s001200000.zip"
            weight = 2.0
            [[version]]
            path = "runs/b.zip"
        """)
        with d:
            c = PoolConfig.load(path)
        self.assertEqual((c.snapshot_every, c.max_size, c.new_weight, c.newer_count_more, c.recency_factor),
                         (50_000, 4, 0.5, True, 2.0))
        self.assertEqual(c.versions, (("runs/a/snapshots/s001200000.zip", 2.0), ("runs/b.zip", 1.0)))

    def test_an_empty_file_is_the_defaults(self):
        d, path = pool_file("")
        with d:
            c = PoolConfig.load(path)
        self.assertEqual((c.snapshot_every, c.max_size, c.new_weight, c.newer_count_more, c.versions),
                         (100_000, 50, 1.0, False, ()))

    def test_refuses_what_would_go_wrong_later(self):
        bad = {
            "a misspelt key": "snapshot_evry = 10",
            "a version without a path": "[[version]]\nweight = 1.0",
            "a version with an unknown key": '[[version]]\npath = "a.zip"\nweigth = 2.0',
            "as many listed as max_size": 'max_size = 1\n[[version]]\npath = "a.zip"',
            "a zero weight": '[[version]]\npath = "a.zip"\nweight = 0.0',
            "a version listed twice": '[[version]]\npath = "a.zip"\n[[version]]\npath = "a.zip"',
            "no snapshots": "snapshot_every = 0",
            "a negative factor": "recency_factor = -1.0",
        }
        for what, text in bad.items():
            d, path = pool_file(text)
            with d, self.subTest(what), self.assertRaises(ValueError):
                PoolConfig.load(path)

    def test_a_listed_file_must_exist(self):
        with tempfile.TemporaryDirectory() as d:
            there = Path(d) / "there.zip"
            there.write_bytes(b"")
            Pool(PoolConfig(versions=((str(there), 1.0),)))
            with self.assertRaises(FileNotFoundError):
                Pool(PoolConfig(versions=((str(Path(d) / "gone.zip"), 1.0),)))


class Weights(unittest.TestCase):
    def test_nothing_listed_starts_from_the_initial_model(self):
        pool = Pool(PoolConfig())
        self.assertTrue(pool.needs_initial)
        with self.assertRaises(RuntimeError):
            pool.draw()
        initial = pool.add("runs/r/snapshots/s000000000.zip", 0)
        self.assertIsNone(initial)
        self.assertFalse(pool.needs_initial)
        self.assertEqual(pool.draw().name, "s000000000")

    def test_the_draw_follows_the_weights(self):
        pool = Pool(PoolConfig(versions=(("a.zip", 3.0),)), rng=random.Random(1), check_files=False)
        pool.add("s1.zip", 100_000)  # new_weight 1.0
        self.assertEqual(sorted(pool.probabilities().values()), [0.25, 0.75])
        counts = collections.Counter(pool.draw().path for _ in range(20_000))
        self.assertAlmostEqual(counts["a.zip"] / 20_000, 0.75, delta=0.02)

    def test_without_the_switch_every_snapshot_weighs_the_same(self):
        pool = Pool(PoolConfig(new_weight=2.0), check_files=False)
        for step in (0, 100_000, 200_000):
            pool.add(f"s{step}.zip", step)
        self.assertEqual([pool.weight(v) for v in pool], [2.0, 2.0, 2.0])

    def test_newer_count_more_compounds_from_one_snapshot_to_the_next(self):
        pool = Pool(PoolConfig(new_weight=1.0, newer_count_more=True, recency_factor=2.0,
                               versions=(("mine.zip", 5.0),)), check_files=False)
        for step in (100_000, 200_000, 300_000):
            pool.add(f"s{step}.zip", step)
        # The newest weighs new_weight, each older one half the next; the listed one keeps its own.
        self.assertEqual([pool.weight(v) for v in pool], [5.0, 0.25, 0.5, 1.0])
        p = pool.probabilities()
        self.assertAlmostEqual(sum(p.values()), 1.0)
        self.assertAlmostEqual(p[pool.versions[0].id], 5.0 / 6.75)


class Snapshots(unittest.TestCase):
    def test_a_snapshot_is_due_every_snapshot_every_steps(self):
        pool = Pool(PoolConfig(snapshot_every=1000), check_files=False)
        self.assertEqual([pool.due(s) for s in (0, 999, 1000)], [False, False, True])
        pool.add("s1.zip", 1024)  # steps come in batches: counted from where it was taken
        self.assertEqual([pool.due(s) for s in (2000, 2023, 2024)], [False, False, True])

    def test_snapshot_paths_sort_by_step(self):
        self.assertEqual(Pool.snapshot_path("runs/sword_a", 1_200_000),
                         Path("runs/sword_a/snapshots/s001200000.zip"))

    def test_above_max_size_a_random_run_snapshot_goes_never_a_listed_one_nor_the_new_one(self):
        listed = (("a.zip", 1.0), ("b.zip", 1.0))
        for seed in range(30):
            pool = Pool(PoolConfig(max_size=4, versions=listed), rng=random.Random(seed), check_files=False)
            self.assertIsNone(pool.add("s1.zip", 1))
            self.assertIsNone(pool.add("s2.zip", 2))
            dropped = pool.add("s3.zip", 3)
            self.assertIn(dropped.path, ("s1.zip", "s2.zip"))
            self.assertEqual(len(pool), 4)
            paths = [v.path for v in pool]
            self.assertEqual(paths[:2], ["a.zip", "b.zip"])
            self.assertIn("s3.zip", paths)
            self.assertNotIn(dropped.path, paths)
            self.assertIsNone(pool.get(dropped.id), "a dropped version is gone by its id too")

    def test_a_drop_changes_no_other_snapshots_weight(self):
        pool = Pool(PoolConfig(max_size=3, newer_count_more=True, recency_factor=2.0),
                    rng=random.Random(0), check_files=False)
        for step in (1, 2, 3):
            pool.add(f"s{step}.zip", step)
        before = {v.path: pool.weight(v) for v in pool}
        dropped = pool.add("s4.zip", 4)
        after = {v.path: pool.weight(v) for v in pool}
        # One step newer for everyone: each kept snapshot halves, whichever was dropped.
        for path, w in after.items():
            if path != "s4.zip":
                self.assertEqual(w, before[path] / 2)
        self.assertEqual(after["s4.zip"], 1.0)
        self.assertNotIn(dropped.path, after)

    def test_ids_are_never_reused(self):
        pool = Pool(PoolConfig(max_size=2), rng=random.Random(0), check_files=False)
        ids = [pool.versions[-1].id for step in range(6) if pool.add(f"s{step}.zip", step) or True]
        self.assertEqual(len(set(ids)), 6)


class Spaces(unittest.TestCase):
    def test_a_version_with_other_spaces_is_refused(self):
        obs = spaces.Box(-1, 1, (12,))
        act = spaces.MultiDiscrete([3, 3, 2])
        check_spaces("same", spaces.Box(-1, 1, (12,)), spaces.MultiDiscrete([3, 3, 2]), obs, act)
        with self.assertRaisesRegex(ValueError, "observation"):
            check_spaces("other obs", spaces.Box(-1, 1, (13,)), act, obs, act)
        with self.assertRaisesRegex(ValueError, "action"):
            check_spaces("other act", obs, spaces.MultiDiscrete([3, 3]), obs, act)


class StatsTest(unittest.TestCase):
    def test_results_per_version_to_csv(self):
        pool = Pool(PoolConfig(versions=(("runs/x/mine.zip", 1.0),)), check_files=False)
        pool.add("runs/r/snapshots/s000100000.zip", 100_000)
        mine, snap = pool.versions
        stats = Stats()
        for v, result in ((mine, "win"), (mine, "win"), (mine, "timeout"), (snap, "loss"), (snap, "draw")):
            stats.record(v, result)
        with self.assertRaises(ValueError):
            stats.record(mine, "won")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "runs" / "r" / "pool_stats.csv"
            stats.write_csv(path)
            rows = list(csv.DictReader(open(path)))
        self.assertEqual(rows, [
            {"version": "mine", "path": "runs/x/mine.zip", "rounds": "3", "wins": "2", "losses": "0",
             "draws": "0", "timeouts": "1", "aborts": "0"},
            {"version": "s000100000", "path": "runs/r/snapshots/s000100000.zip", "rounds": "2", "wins": "0",
             "losses": "1", "draws": "1", "timeouts": "0", "aborts": "0"},
        ])


if __name__ == "__main__":
    unittest.main()
