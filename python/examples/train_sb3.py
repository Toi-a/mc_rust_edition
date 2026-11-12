#!/usr/bin/env python3
"""Self-play PPO with Stable-Baselines3 (`pip install stable-baselines3`), in stages that give
the policy more of the controls each time.

    python python/examples/train_sb3.py --server HOST:PORT -n 16 --stage sword --steps 2000000
    python python/examples/train_sb3.py --local -n 4 --steps 20000          # a local try-out

Stages (`STAGES` below; add your own):
- `sword`: forward, back, jump and attack; the env aims at the opponent for the policy.
- `aim`: the same keys plus strafing and sprint, and the policy turns the camera itself (bins).
- `full`: every key, the hotbar and the camera.

Agents 2k and 2k+1 are each other's opponent and all share one policy, so it plays itself. The
reward is damage dealt minus damage taken. `--local` starts the repo's own offline server and
hands out swords; on your server the kit does that.
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402

from horizon_rl import HorizonVecEnv, rewards  # noqa: E402
from horizon_rl.sb3 import HorizonSB3VecEnv, ReleaseKeys  # noqa: E402

# The action settings of each stage (horizon_rl.actions.ActionCodec). `flat=True`: SB3 takes no
# Dict action space.
STAGES = {
    "sword": dict(camera="aim", buttons=("forward", "back", "jump", "attack"), hotbar=False),
    "aim": dict(camera="bins", hotbar=False,
                buttons=("forward", "back", "left", "right", "jump", "sprint", "attack")),
    "full": dict(camera="bins"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", help="HOST:PORT of an offline-mode 26.3 server")
    ap.add_argument("--local", action="store_true", help="start a local stand-in server instead")
    ap.add_argument("-n", type=int, default=8, help="agents (even: they fight in pairs)")
    ap.add_argument("--stage", default="sword", choices=sorted(STAGES))
    ap.add_argument("--obs", default="state", choices=["state", "image", "both"])
    ap.add_argument("--per-process", type=int, default=1, help="clients per process (less RAM each)")
    ap.add_argument("--steps", type=int, default=1_000_000, help="agent steps in total (20 per agent per second)")
    ap.add_argument("--rollout", type=int, default=512, help="steps per agent between policy updates")
    ap.add_argument("--load", help="a saved model to continue from (same stage)")
    ap.add_argument("--save", default="horizon_ppo", help="where the model is saved at the end")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if not args.server and not args.local:
        ap.error("--server HOST:PORT or --local")

    server = None
    if args.local:
        from horizon_rl.local import LocalServer
        server = LocalServer(tempfile.mkdtemp(prefix="hz-train-", dir="/tmp"))
    env = None
    try:
        env = HorizonVecEnv(server=server.address if server else args.server, n=args.n, pairs=True,
                            obs=args.obs, clients_per_process=args.per_process, flat=True,
                            reward=rewards.duel, **STAGES[args.stage])
        if server:
            for name in env.names:
                server.command(f"give {name} minecraft:diamond_sword")
        print("observation space:", env.single_observation_space)
        print("action space:", env.single_action_space, flush=True)
        venv = VecMonitor(HorizonSB3VecEnv(env))
        policy = {"state": "MlpPolicy", "image": "CnnPolicy", "both": "MultiInputPolicy"}[args.obs]
        if args.load:
            model = PPO.load(args.load, env=venv, device=args.device)
        else:
            model = PPO(policy, venv, n_steps=args.rollout, batch_size=min(1024, args.rollout * args.n),
                        n_epochs=4, gamma=0.99, ent_coef=0.01, verbose=1, device=args.device)
        model.learn(args.steps, callback=ReleaseKeys())
        model.save(args.save)
        print("saved", args.save)
    finally:
        if env is not None:
            env.close()
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
