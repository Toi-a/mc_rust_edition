# Horizon RL

Headless 26.3 clients that Python drives, for PvP self-play. The design, and why it is built this
way: [design.md](design.md).

Status: **M2**: N headless clients from one command or config, each in its own process,
supervised (reconnects, restarts, hang detection), stepped together at 20 Hz from Python. Image
observations (M3) and the Gymnasium env (M4) come next.

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

## Run a fleet

```sh
python/.venv/bin/horizon-rl --server HOST:PORT -n 8 --pairs          # 8 clients, scripted fights
python/.venv/bin/horizon-rl --config python/examples/fleet.toml --policy idle --seconds 600
```

Options (or keys of the TOML file, `FleetConfig` in `python/horizon_rl/fleet.py`): `server`, `n`,
`names` (a format of `{i}`, default `Agent_{i}`), `pairs` (2k and 2k+1 are each other's
opponent) or `target` (a format of `{i}`), `obs` (`state`; images come with M3), `frame_skip`,
`render_distance` (2), `workers` (worker threads per client, 1), `stagger` (seconds between
starts, 0.25), `hang_timeout` (5 s). It prints steps/s, the clients playing, late ticks, ended
episodes and restarts every 10 s. Each client's files and log are under `--root` (default a new
temp folder) in `NAME/`.

From Python (what training uses until the Gymnasium env of M4):

```python
from horizon_rl.fleet import Fleet, FleetConfig

with Fleet(FleetConfig(server="HOST:PORT", n=8, pairs=True)) as fleet:
    fleet.start()                                   # staggered starts, waits for every session
    states, infos = fleet.reset()                   # (n, L) float32; all alive
    while True:
        actions = [(buttons, hotbar, yaw, pitch) for ...]   # one per slot
        states, infos, terminated, truncated, active = fleet.step(actions)
```

- `active[i]` is False while slot `i` is between episodes (dead and respawning, reconnecting, or
  restarting); its action is ignored and its row is zeros. Its next episode's first observation
  comes with `infos[i]["episode_start"]`; resets are automatic.
- `terminated[i]`: it died. `truncated[i]`: a fault cut the episode, named in
  `infos[i]["reset_reason"]`: `crash (…)`, `hang` (no heartbeat for 5 s), `timeout` (no answer
  to a step in 10 s), or `disconnected: <reason>` (the client joins again by itself, after 1, 2, 4
  … 30 s). A crashed or hung process is restarted after 1, 2, 4 … 30 s. Other slots never notice.
- Timing: a headless client waits up to 20 ms for an owed step and moves its tick by the wait
  (`--rl-step-grace`), so N clients come to tick just after Python sends and a batch doesn't miss
  anyone's tick. Measured: 40 clients fighting hold 20 Hz with no late ticks, with or without 10 ms
  of inference per step ([design.md](design.md#m2-measurements)).

## Server

The client speaks only Minecraft 26.3 (protocol 777). On Paper: a 26.3 build, or ViaVersion (plus
ViaBackwards for a newer server). Set `online-mode=false`, and in `bukkit.yml`
`settings.connection-throttle: -1` so many clients from one address may join.

Locally, the repo's own dedicated server stands in (`horizon_rl.local.LocalServer`: a flat,
offline-mode world).

## Tests and benchmarks

```sh
python/.venv/bin/python -m unittest discover -s python/tests -p "test_*.py"   # Python unit tests
python/.venv/bin/python python/tests/e2e_duel.py         # M1 end to end: 2 clients on a local server
python/.venv/bin/python python/tests/e2e_fleet.py        # M2: 4 fighting clients; kill, kick, hang, death
python/.venv/bin/python python/bench/fleet.py --ns 2,8,16,32   # CPU, RSS, 20 Hz as N grows (--infer-ms 10)
python/.venv/bin/python python/bench/transport.py        # binary socket vs the JSON control socket
python3 python/bench/baseline.py . [--fleet 10]          # CPU and RSS per idle client
cargo test -p horizon-rl-proto                           # the wire format
cargo test --profile fast -p horizon-client --lib -- agent::rl   # the fair-play view, kill parsing
```
