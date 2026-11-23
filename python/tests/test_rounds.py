"""Rounds (FleetConfig.rounds): which steps end an episode, and the round reward (no game)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl import rewards  # noqa: E402
from horizon_rl.fleet import episode_end  # noqa: E402

PLAY = {"connected": 1.0, "disconnected": 0.0, "died": 0.0, "round_end": 0.0, "round_abort": 0.0}


def step(**signals):
    return dict(PLAY, **signals)


class EpisodeEnd(unittest.TestCase):
    def test_without_rounds_a_death_ends_the_episode(self):
        self.assertEqual(episode_end(step(died=1.0), rounds=False), (True, False, None))
        self.assertEqual(episode_end(step(), rounds=False), (False, False, None))
        # A round message on a run without rounds changes nothing.
        self.assertEqual(episode_end(step(round_end=1.0), rounds=False), (False, False, None))

    def test_with_rounds_the_round_end_does_and_a_death_alone_doesnt(self):
        self.assertEqual(episode_end(step(died=1.0), rounds=True), (False, False, None))
        for result in ("round_win", "round_loss", "round_draw", "round_timeout"):
            self.assertEqual(episode_end(step(round_end=1.0, **{result: 1.0}), rounds=True),
                             (True, False, None), result)
        # The loser's last step has its death and the round's end together.
        self.assertEqual(episode_end(step(died=1.0, round_end=1.0, round_loss=1.0), rounds=True),
                         (True, False, None))

    def test_an_abort_or_a_lost_connection_truncates(self):
        self.assertEqual(episode_end(step(round_end=1.0, round_abort=1.0), rounds=True), (False, True, "abort"))
        for rounds in (False, True):
            self.assertEqual(episode_end(step(disconnected=1.0), rounds), (False, True, "disconnected"))
            self.assertEqual(episode_end(step(connected=0.0), rounds), (False, True, "disconnected"))


class RoundReward(unittest.TestCase):
    def test_defaults(self):
        r = rewards.Round()
        base = {"target_damage": 0.0, "damage_taken": 0.0}
        self.assertAlmostEqual(r(dict(base, round_win=1.0)), 1.0)
        self.assertAlmostEqual(r(dict(base, round_loss=1.0)), -1.0)
        self.assertAlmostEqual(r(dict(base, round_draw=1.0)), 0.0)
        self.assertAlmostEqual(r(dict(base, round_timeout=1.0)), -1.0)
        self.assertAlmostEqual(r(dict(base, round_abort=1.0)), 0.0)
        # 6 points dealt and 4 taken in a step: 0.06 - 0.04.
        self.assertAlmostEqual(r({"target_damage": 6.0, "damage_taken": 4.0}), 0.02)
        # The winning blow: 5 dealt and the win.
        self.assertAlmostEqual(r({"target_damage": 5.0, "damage_taken": 0.0, "round_win": 1.0}), 1.05)

    def test_every_term_is_a_setting(self):
        r = rewards.Round(win=2.0, timeout=-0.5, dealt=0.1, taken=0.0)
        self.assertAlmostEqual(r({"target_damage": 3.0, "damage_taken": 7.0, "round_win": 1.0}), 2.3)
        self.assertAlmostEqual(r({"target_damage": 0.0, "damage_taken": 0.0, "round_timeout": 1.0}), -0.5)

    def test_hidden_health_deals_nothing(self):
        r = rewards.Round()
        self.assertAlmostEqual(r({"target_damage": float("nan"), "damage_taken": 2.0}), -0.02)

    def test_a_vector_envs_infos(self):
        infos = {"target_damage": np.array([0.0, 4.0, np.nan]), "damage_taken": np.array([4.0, 0.0, 0.0]),
                 "round_win": np.array([0.0, 1.0, 0.0]), "round_loss": np.array([1.0, 0.0, 0.0]),
                 "round_timeout": np.array([0.0, 0.0, 1.0])}
        np.testing.assert_allclose(rewards.Round()(infos), [-1.04, 1.04, -1.0])


if __name__ == "__main__":
    unittest.main()
