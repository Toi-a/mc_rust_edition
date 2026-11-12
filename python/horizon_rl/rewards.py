"""Reward functions: each takes the step's `info` (one agent's dict, or the vector env's dict of
arrays) and returns the reward. The envs give 0 without one; shaping is the trainer's choice."""
import numpy as np


def duel(info):
    """Health points taken from the opponent minus health points lost, as the client sees them
    (`target_damage` is 0 where the server hides the opponent's health)."""
    return np.nan_to_num(info["target_damage"]) - info["damage_taken"]
