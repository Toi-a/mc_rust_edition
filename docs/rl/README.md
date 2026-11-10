# Horizon RL

Headless 26.3 clients that Python drives, for PvP self-play. The design, and why it is built this
way: [design.md](design.md).

Status: **M1** — one headless client controlled from Python (join, observe through the fair-play
view, act, see the result). Several clients from one command (M2), image observations (M3) and
the Gymnasium env (M4) come next.

## Set up

```sh
cargo build --profile fast -p horizon-client --bins                   # target/fast/horizon-client
python3 -m venv python/.venv && python/.venv/bin/pip install -e python/   # numpy, gymnasium
```

The game needs vanilla 26.3 data prepared from your own install
([parity-workflow.md](../parity-workflow.md)), for example from a Prism Launcher instance:

```sh
cargo run --release -p horizon-prepare -- --prepare --prism-dir "PATH/TO/PrismLauncher/instances/INSTANCE" \
    --java-home "PATH/TO/PrismLauncher/java/java-runtime-epsilon"
```

## Run one client

```sh
target/fast/horizon-client --resources .local/horizon-client/26.3 --headless \
    --game-dir "$(mktemp -d)" --server HOST:PORT --name Agent_1 --render-distance 2 \
    --rl-socket /tmp/agent1.sock --rl-shm /dev/shm/agent1 [--rl-target OPPONENT]
```

Run it with fresh `HORIZON_HOME`, `XDG_DATA_HOME` and `XDG_CONFIG_HOME` folders and
`HORIZON_NO_AUDIO=1` (`horizon_rl.local.scratch_env` builds that environment).

```python
from horizon_rl import HorizonClient, proto

with HorizonClient("/tmp/agent1.sock", frame_skip=1) as env:
    state, info = env.reset()                    # waits until the player is alive in the world
    bearing = env.field(state, "target_bearing")  # how far to turn to face the opponent
    yaw = env.field(state, "self_yaw")[0] + bearing[0]
    state, info = env.step(proto.FORWARD | proto.SPRINT | proto.ATTACK, yaw=yaw, pitch=0.0)
    print(info["hit_landed"], info["target_health"], info["late_ticks"])
```

- `step(buttons, hotbar=0, yaw=nan, pitch=nan, frame_skip=0)`: the keys are held (sticky) until a
  later step changes them; `ATTACK` is one click; `yaw`/`pitch` are absolute degrees (NaN keeps).
- `state` is the fair-play view (`env.state_fields` names its parts). `info` holds the reward
  signals, the opponent's health among them; it is never part of `state`.
- After a death the client clicks Respawn itself; `info["died"]` marks the step, and `reset()`
  returns once the player is back.

## Server

The client speaks only Minecraft 26.3 (protocol 777). On Paper: a 26.3 build, or ViaVersion (plus
ViaBackwards for a newer server). Set `online-mode=false`, and in `bukkit.yml`
`settings.connection-throttle: -1` so many clients from one address may join.

Locally, the repo's own dedicated server stands in (`horizon_rl.local.LocalServer`: a flat,
offline-mode world).

## Tests and benchmarks

```sh
python/.venv/bin/python python/tests/e2e_duel.py         # M1 end to end: 2 clients on a local server
python/.venv/bin/python python/bench/transport.py        # binary socket vs the JSON control socket
python3 python/bench/baseline.py . [--fleet 10]          # CPU and RSS per idle client
cargo test -p horizon-rl-proto                           # the wire format
cargo test --profile fast -p horizon-client --lib -- agent::rl   # the fair-play view, kill parsing
```
