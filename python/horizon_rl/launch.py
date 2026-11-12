"""`horizon-rl`: start a fleet of headless clients from one command or config file, and keep it
running (docs/rl/README.md).

    horizon-rl --server HOST:PORT -n 8 --pairs                 # 8 clients, scripted fights
    horizon-rl --config fleet.toml --policy idle --seconds 600

The policy here is for trying a server and the supervisor (`chase`: scripted pairs fighting,
`idle`: standing still). Training drives the same `Fleet` from Python instead.
"""
import argparse
import sys
import time

from .fleet import Fleet, FleetConfig
from .scripted import Chaser, idle


def parse(argv=None):
    p = argparse.ArgumentParser(prog="horizon-rl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="a TOML (or .json) file of FleetConfig settings")
    p.add_argument("--server", help="HOST:PORT of the (offline-mode) server")
    p.add_argument("-n", type=int, help="number of clients")
    p.add_argument("--names", help="player names, a format of {i} (default Agent_{i})")
    p.add_argument("--target", help="each client's opponent, a format of {i}")
    p.add_argument("--pairs", action="store_true", default=None,
                   help="clients 2k and 2k+1 are each other's opponent")
    p.add_argument("--obs", choices=["state"], help="observation mode (images come with M3)")
    p.add_argument("--frame-skip", type=int)
    p.add_argument("--render-distance", type=int)
    p.add_argument("--workers", type=int, help="worker threads per client")
    p.add_argument("--stagger", type=float, help="seconds between process starts")
    p.add_argument("--root", help="folder for the clients' files (default: a new temp folder)")
    p.add_argument("--binary", help="horizon-client (default target/fast/horizon-client)")
    p.add_argument("--resources", help="prepared 26.3 resources (default .local/horizon-client/26.3)")
    p.add_argument("--policy", choices=["chase", "idle"], default="chase")
    p.add_argument("--seconds", type=float, default=0, help="stop after this long (0: until Ctrl-C)")
    return p.parse_args(argv)


def config_of(args):
    overrides = {"server": args.server, "n": args.n, "names": args.names, "target": args.target,
                 "pairs": args.pairs, "obs": args.obs, "frame_skip": args.frame_skip,
                 "render_distance": args.render_distance, "workers": args.workers,
                 "stagger": args.stagger, "root": args.root, "binary": args.binary,
                 "resources": args.resources}
    if args.config:
        return FleetConfig.load(args.config, **overrides)
    if not args.server:
        sys.exit("horizon-rl: --server or --config is needed")
    return FleetConfig(**{k: v for k, v in overrides.items() if v is not None})


def main(argv=None):
    args = parse(argv)
    config = config_of(args)
    with Fleet(config) as fleet:
        print(f"starting {config.n} clients for {config.server} in {fleet.root}", flush=True)
        fleet.start()
        states, _ = fleet.reset()
        if args.policy == "chase":
            policies = [Chaser(fleet.state_fields, seed=i) for i in range(config.n)]
        else:
            policies = [None] * config.n
        states = list(states)
        started = last = time.monotonic()
        steps = late = ended = 0
        while not args.seconds or time.monotonic() - started < args.seconds:
            actions = [p.act(s) if p else idle() for p, s in zip(policies, states)]
            new, infos, term, trunc, active = fleet.step(actions)
            steps += 1
            for i in range(config.n):
                if active[i]:
                    states[i] = new[i]
                    late += int(infos[i].get("late_ticks", 0))
                    ended += int(term[i] or trunc[i])
                    if trunc[i]:
                        print(f"{fleet.slots[i].name}: {infos[i].get('reset_reason')}", flush=True)
            now = time.monotonic()
            if now - last >= 10:
                up = sum(s.state in ("ready", "stepping") for s in fleet.slots)
                restarts = sum(s.restarts for s in fleet.slots)
                print(f"{steps / (now - last):5.1f} steps/s  {up}/{config.n} playing  "
                      f"late ticks {late}  episodes ended {ended}  restarts {restarts}", flush=True)
                last, steps, late, ended = now, 0, 0, 0
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
