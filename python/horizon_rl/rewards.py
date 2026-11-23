"""Reward functions: each takes the step's `info` (one agent's dict, or the vector env's dict of
arrays) and returns the reward. The envs give 0 without one; shaping is the trainer's choice."""
import numpy as np


class Round:
    """The reward of a 1v1 in rounds (`FleetConfig.rounds`): the round's result on the step it
    ends, plus a small term for each health point dealt and taken on every step. Every term is a
    setting, to tune during training:

        reward=rewards.Round(timeout=-0.5, dealt=0.02)

    The defaults: a win +1, a loss -1, a draw (both died on one tick) 0, a timeout -1 for both
    (no passive play), an abort (the opponent left) 0, and +-0.01 per health point dealt and
    taken, so 20 points are worth 0.2, well below a win. Points dealt are 0 where the server hides
    the opponent's health (`target_damage`).
    """

    def __init__(self, win=1.0, loss=-1.0, draw=0.0, timeout=-1.0, abort=0.0, dealt=0.01, taken=0.01):
        self.win, self.loss, self.draw, self.timeout, self.abort = win, loss, draw, timeout, abort
        self.dealt, self.taken = dealt, taken

    def __call__(self, info):
        reward = self.dealt * np.nan_to_num(info["target_damage"]) - self.taken * info["damage_taken"]
        for key, value in (("round_win", self.win), ("round_loss", self.loss), ("round_draw", self.draw),
                           ("round_timeout", self.timeout), ("round_abort", self.abort)):
            reward = reward + value * info.get(key, 0.0)
        return reward

    def __repr__(self):
        return (f"Round(win={self.win}, loss={self.loss}, draw={self.draw}, timeout={self.timeout}, "
                f"abort={self.abort}, dealt={self.dealt}, taken={self.taken})")


def duel(info):
    """Health points taken from the opponent minus health points lost, as the client sees them
    (`target_damage` is 0 where the server hides the opponent's health)."""
    return np.nan_to_num(info["target_damage"]) - info["damage_taken"]
