"""A scripted fighter for tests and benchmarks: it reads only the observation, as a policy would.

It turns to its target, runs at it, strafes a little, jumps now and then, and swings when the
attack indicator is full and the target is in reach. Two of them on each other make a fight
with the movement, chunk loading and combat that training will have.
"""
import math
import random

from . import proto


class Chaser:
    def __init__(self, fields, seed=0, sword_slot=0):
        self.f = fields
        self.rng = random.Random(seed)
        self.strafe = 0
        self.sword_slot = sword_slot

    def _get(self, state, name):
        return state[self.f[name]]

    def act(self, state):
        g = lambda name: self._get(state, name)  # noqa: E731
        hotbar, self.sword_slot = self.sword_slot, 0  # select it once
        if g("target_present")[0] < 0.5:
            # No one to fight: turn slowly and walk, as someone looking around.
            yaw = g("self_yaw")[0] + 9.0
            return (proto.FORWARD, hotbar, yaw, 0.0)
        bearing = g("target_bearing")
        yaw = g("self_yaw")[0] + bearing[0] + self.rng.uniform(-4, 4)
        pitch = g("self_pitch")[0] + bearing[1]
        dist = g("target_distance")[0]
        if self.rng.random() < 0.05:
            self.strafe = self.rng.choice((0, proto.LEFT, proto.RIGHT))
        buttons = self.strafe
        if dist > 2.2:
            buttons |= proto.FORWARD | proto.SPRINT
        if self.rng.random() < 0.03:
            buttons |= proto.JUMP
        if dist < 3.0 and g("self_attack_cooldown")[0] >= 0.95:
            buttons |= proto.ATTACK
        return (buttons, hotbar, yaw, max(-90.0, min(90.0, pitch)))


def idle():
    return (0, 0, math.nan, math.nan)
