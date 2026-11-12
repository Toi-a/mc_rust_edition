"""The Gymnasium envs and their action spaces, over a scripted stand-in for the fleet (no game)."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from horizon_rl import proto  # noqa: E402
from horizon_rl.actions import ActionCodec, from_bits  # noqa: E402
from horizon_rl.env import HorizonEnv, HorizonVecEnv  # noqa: E402
from horizon_rl.fleet import FleetConfig  # noqa: E402
from horizon_rl.parallel import HorizonParallelEnv  # noqa: E402

FIELDS = {"self_pos": slice(0, 3), "self_yaw": slice(3, 4), "self_pitch": slice(4, 5)}
SIGNALS = {"tick": 1.0, "damage_taken": 0.0, "target_damage": 0.0, "died": 0.0}


class FakeFleet:
    """What the envs use of `Fleet`. `script` lists, per step, what each slot does: "play",
    "die", "crash", "away" (between episodes) or "start" (its new episode's first observation).
    A slot's state is `[step, step, step, yaw, pitch]` with the angles the last action set."""

    def __init__(self, n=2, obs="state", script=(), signals=None):
        self.config = FleetConfig(server="h:1", n=n, obs=obs, image_size=(8, 4))
        self.slots = [SimpleNamespace(name=f"Agent_{i}") for i in range(n)]
        self.state_len, self.state_fields, self.info_fields, self.items = 5, FIELDS, {}, []
        self.frames = np.zeros((n, 4, 8), np.uint8) if obs != "state" else None
        self.script = list(script)
        self.signals = signals or {}
        self.sent = []
        self.t = 0
        self.look = np.zeros((n, 2), np.float32)
        self.started = self.closed = False

    def start(self):
        self.started = True
        return self

    def _state(self, i):
        return np.array([self.t] * 3 + list(self.look[i]), np.float32)

    def reset(self, timeout=None):
        self.t += 1
        if self.frames is not None:
            self.frames[:] = self.t
        n = len(self.slots)
        return (np.stack([self._state(i) for i in range(n)]),
                [dict(SIGNALS, episode_start=True) for _ in range(n)])

    def step(self, actions):
        self.sent.append(actions)
        self.t += 1
        n = len(self.slots)
        what = self.script.pop(0) if self.script else ["play"] * n
        states = np.zeros((n, self.state_len), np.float32)
        infos, term, trunc, active = [{} for _ in range(n)], np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        for i, w in enumerate(what):
            if w == "away":
                infos[i] = {"state": "resetting", "restarts": 0}
                continue
            active[i] = True
            if w != "start":
                look = np.array(actions[i][2:], np.float32)
                self.look[i] = np.where(np.isnan(look), self.look[i], look)  # NaN keeps the angle
            states[i] = self._state(i)
            if w == "crash":
                trunc[i] = True
                infos[i] = {"reset_reason": "crash (exit 1)", "restarts": 1}
                continue
            infos[i] = dict(SIGNALS, **self.signals.get((self.t, i), {}))
            if w == "start":
                infos[i]["episode_start"] = True
            if w == "die":
                term[i] = True
                infos[i]["died"] = 1.0
        if self.frames is not None:
            self.frames[:] = self.t
            self.frames[~active] = 0
        return states, infos, term, trunc, active

    def close(self):
        self.closed = True


def with_opponent(fleet):
    """Adds the opponent's fields to the fake's states: slot 0 has one, 40 right and 20 up."""
    fleet.state_len = 8
    fleet.state_fields = dict(FIELDS, target_present=slice(5, 6), target_bearing=slice(6, 8))
    fleet._state = lambda i: np.array([fleet.t] * 3 + [*fleet.look[i], i == 0, 40.0, -20.0], np.float32)
    return fleet


def act(env, **parts):
    """A batch of actions that hold nothing and keep the view, with `parts` replaced."""
    n = env.num_envs
    a = {"buttons": np.zeros((n, 9), np.int8), "hotbar": np.zeros(n, np.int64),
         "look": np.zeros((n, 2), env.single_action_space["look"].dtype)}
    a.update(parts)
    return a


class CodecTest(unittest.TestCase):
    def test_buttons_become_the_wire_bits(self):
        codec = ActionCodec(2)
        buttons = np.zeros((2, 9), np.int8)
        buttons[0, [0, 7]] = 1  # forward, attack
        buttons[1, [4, 5, 8]] = 1  # jump, sprint, use
        bits, hotbar, yaw, pitch = codec.encode(
            {"buttons": buttons, "hotbar": [3, 0], "look": [[90.0, -100.0], [-45.0, 10.0]]},
            yaw=[0.0, 0.0], pitch=[0.0, 0.0])
        self.assertEqual(bits.tolist(), [1 | 128, 16 | 32 | 256])
        self.assertEqual(hotbar.tolist(), [3, 0])
        self.assertEqual(yaw.tolist(), [90.0, -45.0])
        self.assertEqual(pitch.tolist(), [-90.0, 10.0], "pitch stays within straight up and down")

    def test_a_delta_camera_turns_from_where_the_agent_looks(self):
        codec = ActionCodec(1, camera="delta", delta_max=20.0)
        self.assertEqual(codec.space["look"].high.tolist(), [20.0, 20.0])
        _, _, yaw, pitch = codec.encode({"buttons": np.zeros((1, 9)), "hotbar": [0], "look": [[15.0, -50.0]]},
                                        yaw=[170.0], pitch=[-80.0])
        self.assertEqual((yaw[0], pitch[0]), (185.0, -90.0), "the pitch turn is cut to 20, then to -90")

    def test_a_binned_camera_picks_its_turn_from_the_bins(self):
        codec = ActionCodec(1, camera="bins", bins=(-10.0, 0.0, 2.5))
        self.assertEqual(codec.space["look"].nvec.tolist(), [3, 3])
        _, _, yaw, pitch = codec.encode({"buttons": np.zeros((1, 9)), "hotbar": [0], "look": [[2, 0]]},
                                        yaw=[30.0], pitch=[5.0])
        self.assertEqual((yaw[0], pitch[0]), (32.5, -5.0))

    def test_the_turn_cap_takes_the_short_way_round(self):
        codec = ActionCodec(1, max_turn=15.0)
        look = lambda y, p, at: codec.encode(  # noqa: E731
            {"buttons": np.zeros((1, 9)), "hotbar": [0], "look": [[y, p]]}, yaw=[at[0]], pitch=[at[1]])[2:]
        yaw, pitch = look(-170.0, 40.0, at=(170.0, 0.0))
        self.assertEqual((yaw[0], pitch[0]), (185.0, 15.0), "20 degrees through 180, cut to 15")
        yaw, pitch = look(100.0, 0.0, at=(105.0, 3.0))
        self.assertEqual((yaw[0], pitch[0]), (100.0, 0.0), "a turn under the cap is whole")

    def test_the_click_cap_drops_clicks_over_the_rate(self):
        codec = ActionCodec(1, max_cps=2.0)
        click = {"buttons": np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0]]), "hotbar": [0], "look": [[0.0, 0.0]]}
        clicked = [bool(codec.encode(click, [0.0], [0.0])[0][0] & proto.ATTACK) for _ in range(41)]
        # Two clicks per 20 ticks: ticks 0 and 1, then 20 and 21, then 40.
        self.assertEqual([t for t, c in enumerate(clicked) if c], [0, 1, 20, 21, 40])
        self.assertEqual(codec.encode(click, [0.0], [0.0])[0][0] & proto.FORWARD, proto.FORWARD,
                         "the other keys of a dropped click stay held")

    def test_scripted_actions_fit_the_absolute_space(self):
        codec = ActionCodec(1)
        action = from_bits(proto.FORWARD | proto.ATTACK, hotbar=2, yaw=270.0, pitch=95.0)
        self.assertTrue(codec.space.contains(action))
        self.assertEqual((action["buttons"].tolist(), action["look"].tolist()),
                         ([1, 0, 0, 0, 0, 0, 0, 1, 0], [-90.0, 90.0]))


    def test_a_policy_gets_only_the_parts_it_is_given(self):
        codec = ActionCodec(2, camera="aim", buttons=("forward", "back", "jump", "attack"), hotbar=False)
        self.assertEqual(sorted(codec.space.spaces), ["buttons"])
        self.assertEqual(codec.space["buttons"].n, 4)
        bits, hotbar, _, _ = codec.encode({"buttons": [[1, 0, 1, 0], [0, 1, 0, 1]]}, [0.0, 0.0], [0.0, 0.0])
        self.assertEqual((bits.tolist(), hotbar.tolist()), ([1 | 16, 2 | 128], [0, 0]))
        with self.assertRaisesRegex(ValueError, "buttons"):
            ActionCodec(1, buttons=("forward", "fly"))

    def test_aim_assist_faces_the_opponent_within_the_turn_cap(self):
        codec = ActionCodec(3, camera="aim", buttons=("attack",), hotbar=False)
        # The opponent is 30 right and 10 up; 150 left (over the 90 a tick allows); absent.
        _, _, yaw, pitch = codec.encode({"buttons": np.zeros((3, 1))}, yaw=[100.0, 100.0, 100.0],
                                        pitch=[5.0, 5.0, 5.0], aim=[[30.0, -10.0], [-150.0, 60.0], [0.0, 0.0]])
        self.assertEqual(yaw.tolist(), [130.0, 10.0, 100.0])
        self.assertEqual(pitch.tolist(), [-5.0, 65.0, 5.0], "any pitch in one tick")

    def test_by_default_a_tick_turns_90_degrees_of_yaw_and_any_pitch(self):
        absolute = ActionCodec(1)
        _, _, yaw, pitch = absolute.encode({"buttons": np.zeros((1, 9)), "hotbar": [0], "look": [[-170.0, 90.0]]},
                                           yaw=[0.0], pitch=[-90.0])
        self.assertEqual((yaw[0], pitch[0]), (-90.0, 90.0))
        delta = ActionCodec(1, camera="delta")
        self.assertEqual(delta.space["look"].high.tolist(), [90.0, 180.0])
        free = ActionCodec(1, max_turn=0)
        _, _, yaw, _ = free.encode({"buttons": np.zeros((1, 9)), "hotbar": [0], "look": [[-170.0, 0.0]]}, [0.0], [0.0])
        self.assertEqual(yaw[0], -170.0)

    def test_flat_actions_are_one_array(self):
        discrete = ActionCodec(2, camera="bins", buttons=("forward", "attack"), bins=(-10.0, 0.0, 10.0), flat=True)
        self.assertEqual(discrete.space.nvec.tolist(), [2, 2, 10, 3, 3])
        bits, hotbar, yaw, pitch = discrete.encode(np.array([[1, 1, 4, 2, 1], [0, 1, 0, 0, 0]]), [0.0, 0.0], [0.0, 0.0])
        self.assertEqual((bits.tolist(), hotbar.tolist(), yaw.tolist(), pitch.tolist()),
                         ([1 | 128, 128], [4, 0], [10.0, -10.0], [0.0, -10.0]))
        aimed = ActionCodec(1, camera="aim", buttons=("forward", "back", "jump", "attack"), hotbar=False, flat=True)
        self.assertEqual(aimed.space.nvec.tolist(), [2, 2, 2, 2])
        continuous = ActionCodec(1, camera="delta", buttons=("forward", "attack"), hotbar=False, flat=True)
        self.assertEqual(continuous.space.shape, (4,))
        bits, _, yaw, pitch = continuous.encode(np.array([0.3, -0.2, 0.5, -0.25], np.float32), [10.0], [0.0])
        self.assertEqual((bits.tolist(), yaw.tolist(), pitch.tolist()), ([1], [55.0], [-45.0]),
                         "above 0 holds a key; the look spans the turn a tick allows")
        with self.assertRaisesRegex(ValueError, "hotbar"):
            ActionCodec(1, camera="delta", flat=True)
        for codec in (discrete, aimed, continuous, ActionCodec(1)):
            self.assertTrue(codec.space.contains(codec.placeholder()))


class VecEnvTest(unittest.TestCase):
    def test_aim_assist_reads_the_bearing_of_the_observation(self):
        fleet = with_opponent(FakeFleet(2))
        env = HorizonVecEnv(fleet=fleet, camera="aim", buttons=("forward", "attack"), hotbar=False, flat=True)
        env.reset()
        env.step(np.array([[1, 1], [1, 0]]))
        self.assertEqual(fleet.sent[-1], [(1 | 128, 0, 40.0, -20.0), (1, 0, 0.0, 0.0)],
                         "the one with an opponent turns to it; the other keeps its view")

    def test_spaces_follow_the_observation_mode(self):
        for obs, shape in (("state", (3, 5)), ("image", (3, 4, 8))):
            env = HorizonVecEnv(fleet=FakeFleet(3, obs))
            first, _ = env.reset()
            self.assertEqual(first.shape, shape)
            self.assertTrue(env.observation_space.contains(first))
            nxt = env.step(env.action_space.sample())[0]
            self.assertTrue(env.observation_space.contains(nxt))
        env = HorizonVecEnv(fleet=FakeFleet(3, "both"))
        first, _ = env.reset()
        self.assertEqual((first["state"].shape, first["image"].shape, first["image"].dtype),
                         ((3, 5), (3, 4, 8), np.uint8))
        self.assertTrue(env.observation_space.contains(first))

    def test_an_ended_agent_is_no_transition_until_its_new_episode(self):
        fleet = FakeFleet(2, "both", script=[["die", "play"], ["away", "play"], ["start", "play"],
                                             ["play", "play"]])
        env = HorizonVecEnv(fleet=fleet, reward=lambda info: info["tick"] * 0 + 1.0)
        _, infos = env.reset()
        self.assertEqual(infos["valid"].tolist(), [False, False], "a reset is no transition")
        obs, rewards, term, trunc, infos = env.step(act(env))
        self.assertEqual((term.tolist(), infos["valid"].tolist(), rewards.tolist()),
                         ([True, False], [True, True], [1.0, 1.0]), "the death is the episode's last transition")
        died_at = obs["state"][0].copy()
        obs, rewards, term, trunc, infos = env.step(act(env))
        self.assertEqual((infos["active"].tolist(), infos["valid"].tolist(), rewards.tolist()),
                         ([False, True], [False, True], [0.0, 1.0]))
        self.assertEqual(obs["state"][0].tolist(), died_at.tolist(), "its last observation stands")
        self.assertEqual(int(obs["image"][0, 0, 0]), 2, "and its last frame")
        self.assertEqual((term.tolist(), trunc.tolist()), ([False, False], [False, False]))
        obs, rewards, term, trunc, infos = env.step(act(env))
        self.assertEqual((infos["active"].tolist(), infos["episode_start"].tolist(), infos["valid"].tolist()),
                         ([True, True], [True, False], [False, True]))
        self.assertEqual((rewards.tolist(), obs["state"][0, 0]), ([0.0, 1.0], 4.0),
                         "the first observation of the new episode, without a reward")
        obs, rewards, term, trunc, infos = env.step(act(env))
        self.assertEqual((infos["valid"].tolist(), rewards.tolist()), ([True, True], [1.0, 1.0]))

    def test_a_fault_truncates_and_names_its_reason(self):
        env = HorizonVecEnv(fleet=FakeFleet(2, script=[["play", "crash"]]),
                            reward=lambda info: info["target_damage"] - info["damage_taken"])
        env.reset()
        _, rewards, term, trunc, infos = env.step(act(env))
        self.assertEqual((term.tolist(), trunc.tolist()), ([False, False], [False, True]))
        self.assertEqual((infos["reset_reason"][1], infos["_reset_reason"].tolist()),
                         ("crash (exit 1)", [False, True]))
        self.assertEqual(rewards.tolist(), [0.0, 0.0])

    def test_rewards_come_from_the_infos_signals(self):
        fleet = FakeFleet(2, signals={(2, 0): {"target_damage": 6.0}, (2, 1): {"damage_taken": 6.0}})
        from horizon_rl.rewards import duel
        env = HorizonVecEnv(fleet=fleet, reward=duel)
        env.reset()
        _, rewards, _, _, infos = env.step(act(env))
        self.assertEqual(rewards.tolist(), [6.0, -6.0])
        self.assertEqual(infos["damage_taken"].tolist(), [0.0, 6.0])

    def test_camera_turns_add_to_the_observed_angles(self):
        fleet = FakeFleet(2)
        env = HorizonVecEnv(fleet=fleet, camera="delta", delta_max=30.0)
        env.reset()
        turn = act(env, look=np.array([[10.0, -5.0], [-30.0, 0.0]], np.float32))
        env.step(turn)
        obs = env.step(turn)[0]
        self.assertEqual(obs[:, 3:].tolist(), [[20.0, -10.0], [-60.0, 0.0]])
        self.assertEqual(fleet.sent[-1][0], (0, 0, 20.0, -10.0))

    def test_a_release_lets_go_of_the_keys_and_keeps_what_ended(self):
        fleet = FakeFleet(2, script=[["play", "play"], ["die", "play"], ["away", "play"]])
        env = HorizonVecEnv(fleet=fleet)
        env.reset()
        held = act(env, buttons=np.ones((2, 9), np.int8))
        env.step(held)
        env.release()
        buttons, hotbar, yaw, pitch = fleet.sent[-1][0]
        self.assertEqual((buttons, hotbar, np.isnan(yaw), np.isnan(pitch)), (0, 0, True, True))
        _, _, term, _, infos = env.step(held)
        self.assertEqual((term.tolist(), infos["valid"].tolist()), ([True, False], [False, True]),
                         "the death during the release is reported, as no transition")
        self.assertFalse(env.step(held)[2].any())

    def test_closing_closes_the_fleet(self):
        fleet = FakeFleet(1)
        env = HorizonVecEnv(fleet=fleet)
        self.assertTrue(fleet.started)
        env.close()
        self.assertTrue(fleet.closed)


class SingleEnvTest(unittest.TestCase):
    def test_one_agent_steps_and_ends_with_its_death(self):
        fleet = FakeFleet(1, "both", script=[["play"], ["die"]], signals={(2, 0): {"target_damage": 2.0}})
        env = HorizonEnv(fleet=fleet, reward=lambda info: info["target_damage"])
        obs, info = env.reset()
        self.assertTrue(env.observation_space.contains(obs))
        self.assertTrue(info["episode_start"])
        action = env.action_space.sample()
        action["buttons"][:] = [1, 0, 0, 0, 0, 1, 0, 0, 0]
        obs, reward, term, trunc, info = env.step(action)
        self.assertEqual((reward, term, trunc, fleet.sent[0][0][0]), (2.0, False, False, 1 | 32))
        self.assertTrue(env.observation_space.contains(obs))
        obs, reward, term, trunc, info = env.step(action)
        self.assertEqual((term, trunc, info["died"]), (True, False, 1.0))
        obs, info = env.reset()
        self.assertEqual(obs["state"][0], 4.0)

    def test_it_is_one_agent(self):
        with self.assertRaisesRegex(ValueError, "one agent"):
            HorizonEnv(fleet=FakeFleet(2))


class ParallelEnvTest(unittest.TestCase):
    def test_agents_leave_when_their_episode_ends_and_come_back(self):
        fleet = FakeFleet(2, script=[["die", "play"], ["away", "play"], ["start", "play"]])
        env = HorizonParallelEnv(vec_env=HorizonVecEnv(fleet=fleet, camera="bins"))
        obs, infos = env.reset()
        self.assertEqual((env.agents, sorted(obs)), (["Agent_0", "Agent_1"], ["Agent_0", "Agent_1"]))
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, term, trunc, infos = env.step(actions)
        self.assertEqual((term, env.agents), ({"Agent_0": True, "Agent_1": False}, ["Agent_1"]))
        obs, rewards, term, trunc, infos = env.step({"Agent_1": actions["Agent_1"]})
        self.assertEqual(sorted(obs), ["Agent_1"])
        obs, rewards, term, trunc, infos = env.step({"Agent_1": actions["Agent_1"]})
        self.assertEqual((env.agents, infos["Agent_0"]["episode_start"]), (["Agent_0", "Agent_1"], True))
        self.assertNotIn("reset_reason", infos["Agent_0"])


try:
    from horizon_rl.sb3 import HorizonSB3VecEnv
except ImportError:
    HorizonSB3VecEnv = None


@unittest.skipIf(HorizonSB3VecEnv is None, "stable-baselines3 isn't installed")
class SB3Test(unittest.TestCase):
    def make(self, obs="both", **kw):
        fleet = with_opponent(FakeFleet(2, obs, **kw))
        return HorizonSB3VecEnv(HorizonVecEnv(fleet=fleet, camera="aim", hotbar=False, flat=True,
                                              buttons=("forward", "back", "jump", "attack"),
                                              reward=lambda info: info["target_damage"])), fleet

    def test_it_speaks_sb3s_vec_env(self):
        env, fleet = self.make(script=[["die", "play"], ["away", "crash"]], signals={(2, 1): {"target_damage": 3.0}})
        obs = env.reset()
        self.assertEqual((obs["image"].shape, env.observation_space["image"].shape), ((2, 1, 4, 8), (1, 4, 8)))
        obs, rewards, dones, infos = env.step(np.array([[1, 0, 0, 1], [0, 0, 0, 0]]))
        self.assertEqual((rewards.tolist(), dones.tolist(), fleet.sent[-1][0][0]), ([0.0, 3.0], [True, False], 1 | 128))
        self.assertEqual((infos[0]["TimeLimit.truncated"], infos[0]["terminal_observation"]["state"].shape), (False, (8,)))
        self.assertNotIn("terminal_observation", infos[1])
        obs, rewards, dones, infos = env.step(np.zeros((2, 4), int))
        self.assertEqual((dones.tolist(), infos[0]["valid"], infos[1]["TimeLimit.truncated"], infos[1]["reset_reason"]),
                         ([False, True], False, True, "crash (exit 1)"))

    def test_a_dict_action_space_is_refused(self):
        with self.assertRaisesRegex(ValueError, "flat=True"):
            HorizonSB3VecEnv(HorizonVecEnv(fleet=FakeFleet(1)))

    def test_ppo_trains_on_it(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecMonitor
        env, fleet = self.make(obs="state")
        from horizon_rl.sb3 import ReleaseKeys
        PPO("MlpPolicy", VecMonitor(env), n_steps=8, batch_size=8, n_epochs=1, device="cpu").learn(
            32, callback=ReleaseKeys())
        released = [a for a in fleet.sent if np.isnan(a[0][2])]
        self.assertEqual((len(fleet.sent) - len(released), len(released)), (16, 2),
                         "two rollouts of 8 steps, the keys released after each")


if __name__ == "__main__":
    unittest.main()
