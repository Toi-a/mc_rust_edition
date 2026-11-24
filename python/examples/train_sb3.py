#!/usr/bin/env python3
"""Self-play PPO with Stable-Baselines3 (`pip install stable-baselines3`), in stages that give
the policy more of the controls each time.

    python python/examples/train_sb3.py --server HOST:PORT -n 16 --stage sword --steps 2000000
    python python/examples/train_sb3.py --local -n 4 --steps 20000          # a local try-out

Stages (`STAGES` below; add your own):
- `sword`: forward, back, jump and attack; the env aims at the opponent for the policy. It
  observes 16 normalised numbers, the opponent's health among them (a server that shows it).
- `aim`: the same keys plus strafing and sprint, and the policy turns the camera itself (bins).
- `full`: every key, the hotbar and the camera.

Agents 2k and 2k+1 are each other's opponent and all share one policy, so it plays itself. The
reward is damage dealt minus damage taken. `--local` starts the repo's own offline server and
hands out swords; on your server the kit does that.

`--rounds`: the server's plugin runs 1v1 rounds (docs/rl/design.md#rounds-and-the-opponent-pool-m6-planned).
An episode is then a round, the reward is `rewards.Round` (the result, plus a little per health
point dealt and taken), and the policy also observes the round's time left. The local stand-in
server has no rounds.

`--pool FILE --run NAME`: the learner fights frozen past versions of itself instead of its
current self (docs/rl/design.md#the-opponent-pool). Agents 2k are the learner, 2k+1 its
opponents, drawn from the pool file's versions and the run's snapshots
(`runs/NAME/snapshots/`); the results per version go to `runs/NAME/pool_stats.csv` (and
TensorBoard in `runs/NAME/tb` when it is installed), the model to `runs/NAME/final.zip`.

    python python/examples/train_sb3.py --server HOST:PORT -n 16 --rounds --pool pool.toml --run sword_a
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import VecMonitor  # noqa: E402

from horizon_rl import HorizonVecEnv, rewards  # noqa: E402
from horizon_rl.observations import PRESETS  # noqa: E402
from horizon_rl.sb3 import HorizonSB3VecEnv, ReleaseKeys  # noqa: E402

# The action settings of each stage (horizon_rl.actions.ActionCodec) and what it observes
# (horizon_rl.observations; none given: the game's whole state vector, raw). `flat=True`: SB3
# takes no Dict action space.
STAGES = {
    "sword": dict(camera="aim", buttons=("forward", "back", "jump", "attack"), hotbar=False,
                  observation="sword"),  # a small normalised view, with the opponent's health
    "aim": dict(camera="bins", hotbar=False,
                buttons=("forward", "back", "left", "right", "jump", "sprint", "attack")),
    "full": dict(camera="bins"),
}


def stage_settings(stage, rounds):
    """A stage's env settings; with rounds the chosen observation also has the round's time left
    (the whole state vector has it already)."""
    settings = dict(STAGES[stage])
    observation = settings.get("observation")
    if rounds and observation is not None:
        names = PRESETS[observation] if isinstance(observation, str) else observation
        settings["observation"] = tuple(names) + ("round_time_left",)
    return settings


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
    ap.add_argument("--login-slot", "--hotbar", type=int, default=1, dest="login_slot",
                    help="the hotbar slot 1..9 each agent holds at login (the kit's sword: 1)")
    ap.add_argument("--rounds", action="store_true", help="the server's plugin runs 1v1 rounds")
    ap.add_argument("--pool", help="a pool file (TOML): fight frozen past versions, not the current self")
    ap.add_argument("--run", help="the run's name: its snapshots, stats and model go to runs/NAME/")
    ap.add_argument("--opponent-device", default="cpu", help="where the pool's versions run")
    args = ap.parse_args()
    if args.pool and not args.run:
        ap.error("--pool needs --run NAME (where its snapshots go)")
    if args.pool and not args.rounds:
        print("note: without --rounds an episode ends at a death and no results are counted per version")
    if not args.server and not args.local:
        ap.error("--server HOST:PORT or --local")
    if args.rounds and args.local:
        ap.error("--rounds needs the round plugin: the local stand-in server has none")

    server = None
    if args.local:
        from horizon_rl.local import LocalServer
        server = LocalServer(tempfile.mkdtemp(prefix="hz-train-", dir="/tmp"))
    env = None
    try:
        env = HorizonVecEnv(server=server.address if server else args.server, n=args.n, pairs=True,
                            obs=args.obs, clients_per_process=args.per_process, flat=True,
                            rounds=args.rounds, reward=rewards.Round() if args.rounds else rewards.duel,
                            login_slot=args.login_slot, **stage_settings(args.stage, args.rounds))
        if server:
            for name in env.names:
                server.command(f"give {name} minecraft:diamond_sword")
        print("observation space:", env.single_observation_space)
        print("action space:", env.single_action_space, flush=True)
        venv = HorizonSB3VecEnv(env)
        callbacks = [ReleaseKeys()]
        run_dir = Path("runs") / args.run if args.run else None
        tensorboard = None
        if args.pool:
            from horizon_rl.pool import Pool, PoolConfig
            from horizon_rl.pool_sb3 import OpponentPoolVecEnv, PoolSnapshots, PoolStats
            pool = Pool(PoolConfig.load(args.pool))
            venv = OpponentPoolVecEnv(venv, pool, opponent_device=args.opponent_device)
            callbacks += [PoolSnapshots(run_dir), PoolStats(run_dir)]
            print(f"pool: {len(pool)} versions listed, {venv.num_envs} learner seats", flush=True)
        if run_dir is not None:
            try:
                import tensorboard  # noqa: F401
                tensorboard = str(run_dir / "tb")
            except ImportError:
                pass
        venv = VecMonitor(venv)
        policy = {"state": "MlpPolicy", "image": "CnnPolicy", "both": "MultiInputPolicy"}[args.obs]
        if args.load:
            model = PPO.load(args.load, env=venv, device=args.device, tensorboard_log=tensorboard)
        else:
            model = PPO(policy, venv, n_steps=args.rollout, batch_size=min(1024, args.rollout * venv.num_envs),
                        n_epochs=4, gamma=0.99, ent_coef=0.01, verbose=1, device=args.device,
                        tensorboard_log=tensorboard)
        # A loaded model counts its steps on, so its snapshots don't take earlier ones' names.
        model.learn(args.steps, callback=callbacks, reset_num_timesteps=not args.load)
        save = run_dir / "final.zip" if run_dir is not None and args.save == ap.get_default("save") else args.save
        model.save(save)
        print("saved", save)
    finally:
        if env is not None:
            env.close()
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
