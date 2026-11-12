# Horizon RL

Headless 26.3 clients that Python drives, for PvP self-play. The design, and why it is built this
way: [design.md](design.md).

Status: **M3**: N headless clients from one command or config, each in its own process,
supervised (reconnects, restarts, hang detection), stepped together at 20 Hz from Python, with
the fair-play state vector and, optionally, grayscale frames of each player's view. The
Gymnasium env (M4) comes next.

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

### Images

Add `--rl-image WxH` (W a multiple of 4, sides 8 to 512): the headless client then also draws the
player's view offscreen, with no display, and each observation carries its grayscale frame:

```python
with HorizonClient("/tmp/agent1.sock") as env:
    state, info = env.reset()
    state, info = env.step(proto.FORWARD, yaw=90.0, pitch=0.0)
    frame = env.frame()            # (H, W) uint8, rows top first; env.frame_shape
    info["frame_fresh"], info["frame_ms"]
```

- The frame is drawn after the step's last tick, so it shows the same tick as `state`: one frame
  per observation (with frame skip 4, one per 4 ticks), nothing drawn in between.
- It is vanilla's view: no HUD, hand or name tags by default (F1); `--rl-hud` draws them. Nothing
  else is ever drawn. Luma is Rec. 601 (`0.299 R + 0.587 G + 0.114 B`) of the displayed picture.
- Cheap settings for it go in the game folder's `options.txt` (`horizon_rl.local.RENDER_OPTIONS`,
  written by the fleet: no clouds, minimal particles, no smooth lighting or mipmaps).
- On NVIDIA without Resizable BAR, run image clients with `HORIZON_NO_BAR_UPLOADS=1` (the fleet
  does): otherwise each process maps ~20 MB of the GPU's 256 MB BAR1 window, and from about a
  dozen clients on, new ones can't open the GPU.
- `HORIZON_SUBMIT_THREAD=0` (the fleet sets it) saves a thread: each frame is read back at once
  anyway.

## Run a fleet

```sh
python/.venv/bin/horizon-rl --server HOST:PORT -n 8 --pairs          # 8 clients, scripted fights
python/.venv/bin/horizon-rl --config python/examples/fleet.toml --policy idle --seconds 600
```

Options (or keys of the TOML file, `FleetConfig` in `python/horizon_rl/fleet.py`): `server`, `n`,
`names` (a format of `{i}`, default `Agent_{i}`), `pairs` (2k and 2k+1 are each other's
opponent) or `target` (a format of `{i}`), `obs` (`state`, `image` or `both`: every mode returns
the state vector; `image` and `both` add frames), `image_size` (`[64, 64]`), `hud` (false), `frame_skip`,
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

- With images, `fleet.frames` is an `(n, H, W)` uint8 array of the same observations, after each
  `reset()` and `step()` (zeros for an inactive slot).
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
python/.venv/bin/python python/tests/e2e_image.py        # M3: frames: sky/ground, the opponent, frame skip, HUD
python/.venv/bin/python python/bench/fleet.py --ns 2,8,16,32   # CPU, RSS, 20 Hz as N grows (--infer-ms 10)
python/.venv/bin/python python/bench/fleet.py --ns 8,32 --obs image --image 64x64   # the same with frames (+ GPU memory)
python/.venv/bin/python python/bench/transport.py        # binary socket vs the JSON control socket
python3 python/bench/baseline.py . [--fleet 10]          # CPU and RSS per idle client
cargo test -p horizon-rl-proto                           # the wire format
cargo test -p horizon-client-render --lib gfx::luma      # the luma pass on the GPU (skips without one)
cargo test --profile fast -p horizon-client --lib -- agent::rl   # the fair-play view, kill parsing
```
