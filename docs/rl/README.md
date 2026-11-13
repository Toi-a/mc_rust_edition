# Horizon RL

Headless 26.3 clients that Python drives, for PvP self-play. The design, and why it is built this
way: [design.md](design.md).

Status: **M4**: N headless clients from one command or config, each in its own process,
supervised (reconnects, restarts, hang detection), stepped together at 20 Hz from Python, with
the fair-play state vector and, optionally, grayscale frames of each player's view; optionally
several clients per process (model B); and [Gymnasium environments](#gymnasium) over them.
Removing what this fork doesn't need (M5) comes next.

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
  signals, the opponent's health among them; it is part of `state` only with `--rl-target-health`.
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
- `HORIZON_SECTION_ARENA_VERTICES` (the fleet sets 262144) sizes the terrain buffer's first page
  for a small view: ~7.5 MB of GPU memory instead of ~30 MB; it grows when needed.
- `HORIZON_LEAN_ATLAS=1` (the fleet sets it unless `--hud`) leaves the GUI's textures and the
  menus' slots out of the entity/item/GUI atlas: 31 MB of GPU memory instead of 96. Every sprite
  keeps its name; those left out draw nothing. At most 16 maps (in frames or hands) draw at once.
- `HORIZON_MESH_POOL_BYTES` (the fleet sets 1 MB) caps the cleared mesh buffers the meshing
  workers keep (16 MB by default).
- With several clients per process (`--per-process K`), they share the first client's textures
  (block textures, and the entity/item/GUI atlas unless `--hud`): an image client then takes
  ~31 MB of GPU memory instead of ~87 MB ([design](design.md#m5-measurements)).
- A state-only client loads as a server does: no models, no texture pixels, of the GUI only the
  font's metrics. It takes ~125 MB of RAM alone and ~45 MB at `--per-process 8`.

## Run a fleet

```sh
python/.venv/bin/horizon-rl --server HOST:PORT -n 8 --pairs          # 8 clients, scripted fights
python/.venv/bin/horizon-rl --config python/examples/fleet.toml --policy idle --seconds 600
```

Options (or keys of the TOML file, `FleetConfig` in `python/horizon_rl/fleet.py`): `server`, `n`,
`names` (a format of `{i}`, default `Agent_{i}`), `pairs` (2k and 2k+1 are each other's
opponent) or `target` (a format of `{i}`), `obs` (`state`, `image` or `both`: every mode returns
the state vector; `image` and `both` add frames), `image_size` (`[64, 64]`), `hud` (false), `target_health` (false: the
server shows players' health, so the opponent's is observed), `frame_skip`,
`render_distance` (2), `workers` (worker threads per client, 1), `clients_per_process` (1; see
below), `stagger` (seconds between starts, 0.25), `hang_timeout` (5 s). It prints steps/s, the clients playing, late ticks, ended
episodes and restarts every 10 s. Each client's files and log are under `--root` (default a new
temp folder) in `NAME/`.

From Python, without Gymnasium (the layer the [Gymnasium envs](#gymnasium) are built on):

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

### Several clients per process

`clients_per_process = K` (`horizon-rl --per-process K`) runs slots `iK … iK+K-1` in one process
(`--rl-client`, process model B): K clients share one copy of the game's resources and, with
images, one GPU device, each on its own thread. It saves memory per client
([design.md](design.md#model-b-measurements)); the price is fault isolation: a crash or hang of
one restarts the K together (each a truncated episode; the others' `reset_reason` ends in
`(shard)`). A kick still stays one client's own. Every slot keeps its own socket, shared memory and
folder (`NAME/`); the process's log is in its first slot's folder.

## Gymnasium

```python
from horizon_rl import HorizonVecEnv, rewards

env = HorizonVecEnv(server="HOST:PORT", n=8, pairs=True, obs="both", camera="delta",
                    reward=rewards.duel)           # starts the 8 clients; frame skip 1: a step is a tick
obs, infos = env.reset()                           # all alive in the world
while training:
    actions = policy(obs)                          # or env.action_space.sample()
    obs, rewards_, terminated, truncated, infos = env.step(actions)
    valid = infos["valid"]                         # train on these rows only (below)
env.close()
```

`python/examples/random_agent.py --local -n 4` runs this loop with random actions on a local
stand-in server (`--server HOST:PORT` for a real one).

- **`HorizonVecEnv`** (`gymnasium.vector.VectorEnv`): every `FleetConfig` setting as a keyword
  (or `HorizonVecEnv(FleetConfig(...))`, or a TOML file through `FleetConfig.load`), so `obs`,
  `image_size`, `frame_skip`, `clients_per_process` and the rest work as for a fleet. One `step`
  sends all actions and waits for all observations: one game tick with frame skip 1, so 20 steps
  per second for every agent, in one Python thread.
- **`HorizonEnv`** (`gymnasium.Env`): one agent (`HorizonEnv(server=..., names="Agent",
  target="Rival")`). After `terminated` or `truncated`, `reset()` waits until the agent is alive
  in the world again (respawned, rejoined or restarted).
- **Observation space**, by `obs`: `"state"`: `Box(-inf, inf, (L,), float32)`; `"image"`:
  `Box(0, 255, (H, W), uint8)`; `"both"`: a `Dict` with `state` and `image`. The state is chosen
  per run: [Observations](#observations).
- **Action space**, built from parts so a first policy can start small and later ones get
  more (`horizon_rl.actions.ActionCodec`). By default
  `Dict(buttons=MultiBinary(9), hotbar=Discrete(10), look=...)`:
  - `buttons=`: the keys the policy has, a subset of `BUTTON_NAMES` (forward, back, left, right,
    jump, sprint, sneak, attack, use), e.g. `buttons=("forward", "back", "jump", "attack")`. Held
    while 1, except `attack`, one click per step. Keys left out are never pressed.
  - `hotbar=False` drops the hotbar (otherwise 0 keeps the slot, 1..9 select).
  - `camera=`: `"absolute"` (default): `Box([-180, -90], [180, 90])`, the yaw and pitch to face;
    `"delta"`: a `Box` of turns in degrees per tick (`delta_max=`, by default what the turn cap
    allows); `"bins"`: `MultiDiscrete([B, B])`, indices into `bins=` (turns in degrees; default
    -45, -15, -5, -1.5, 0, 1.5, 5, 15, 45); `"aim"`: no `look`, the env faces the opponent for
    the policy. Turns are added to the yaw and pitch of the last observation, in Python; the
    game still gets absolute angles.
  - `flat=True`: one array per action instead of a dict, for trainers without `Dict` action
    spaces (Stable-Baselines3): a `MultiDiscrete` (one column per button, then the hotbar, then
    the yaw and pitch bins) with the `"bins"` and `"aim"` cameras; a `Box(-1, 1)` with the
    continuous ones (a key is held while its value is above 0, the look spans the allowed turn;
    no hotbar).
- **`camera="aim"` is aim assist**: a training aid for a first stage, not fair play. It uses only
  the opponent's bearing from the observation, and obeys the turn cap.
- **Human caps**: `max_turn=(yaw, pitch)`, the largest turn per tick in degrees, 0 for no cap
  on an axis. Default `(90, 0)`: at most 90 degrees left or right per tick, any pitch; an absolute
  target further away is approached by the short way. `max_cps=` (default 0, off): attack clicks
  per second; a click over the rate is dropped, the other keys of that action stay.
- **Reward**: 0 unless `reward=` is given: a function of the step's `infos` returning the
  rewards. `horizon_rl.rewards.duel` is damage dealt minus damage taken, as the client sees them.
  Every signal is in `infos` ([design.md](design.md#rewards-episodes-and-info)), each as an
  `(n,)` array with Gymnasium's `_key` mask of the agents that have it.
- **Episodes** (vector env): Gymnasium's next-step autoreset (`metadata["autoreset_mode"]`),
  stretched, because a reset takes game time and the other agents can't wait for it. The step
  that ends agent `i`'s episode has `terminated[i]` (it died) or `truncated[i]` (a fault:
  `infos["reset_reason"][i]`). Then, for the steps its reset takes (a respawn: about a second;
  a restart: several), `infos["active"][i]` is False: its action is ignored, its observation is
  its last one and its reward 0. The step that returns its new episode's first observation has
  `infos["episode_start"][i]`. **`infos["valid"][i]`** is True exactly when row `i` is a real
  transition (the agent acted on its previous observation and this is what followed): put only
  those rows in a replay buffer or rollout, and cut the trajectory at `episode_start`.
- **Policy updates**: the game doesn't wait while a trainer computes. Call `env.release()`
  before a pause of more than a tick or two (a PPO update): it lets go of every agent's keys, so
  they stand still instead of running on with their last action.
- **Stable-Baselines3** (optional, `pip install stable-baselines3`):
  `horizon_rl.sb3.HorizonSB3VecEnv(env)` is SB3's `VecEnv` over a `flat=True` env, and
  `ReleaseKeys()` the callback that calls `release()` after each rollout.
  `python/examples/train_sb3.py --local -n 4 --steps 20000` trains PPO in self-play (agents in
  pairs, one shared policy), in stages: `--stage sword` (forward, back, jump, attack, aim
  assist), `aim` (plus strafing and sprint, camera by bins), `full`. SB3 can't skip rows, so the
  ~20 steps an agent waits for its respawn are stored as steps of its next episode with reward
  0 (`info["valid"]` False).
- **PettingZoo** (optional, `pip install pettingzoo`): `horizon_rl.parallel.HorizonParallelEnv`
  is a `ParallelEnv` over the same vector env, with the players' names as agents. An agent whose
  episode ended leaves `agents` and comes back by itself when its next one starts.

### Observations

Without `observation=`, the state is the game's whole vector, raw: 70 values,
`env.state_fields` (name -> slice), `env.items` the item registry its item indices refer to.
With it, the env returns the features you list, in that order (`env.observation_fields` says
where each is):

```python
env = HorizonVecEnv(server=..., n=8, pairs=True,
                    observation=("self_vel_local", "self_health", "self_attack_cooldown",
                                 "target_present", "target_distance", "target_health"))
env = HorizonVecEnv(server=..., n=8, pairs=True, observation="sword")   # a preset
```

**Features** (`horizon_rl.observations`). Every field of the game's state is one, under its
own name:

| Feature | Values | Divided by (with `normalize=True`, the default) |
|---|---|---|
| `self_pos` | 3 | – (world coordinates) |
| `self_vel` | 3 | 0.5 blocks per tick |
| `self_yaw`, `self_pitch` | 1, 1 | 180 (after folding into -180..180), 90 |
| `self_health`, `self_absorption`, `self_food`, `self_armor` | 1 each | 20 |
| `self_attack_cooldown` | 1 | – (0..1, 1 = charged) |
| `self_selected` | 1 | 8 |
| `self_hotbar_item`, `self_hotbar_count` | 9, 9 | – (item index + 1), 64 |
| `self_offhand_item`, `self_offhand_count` | 1, 1 | –, 64 |
| `self_armor_item` | 4 | – |
| `self_on_ground`, `self_sprinting`, `self_sneaking`, `self_using_item`, `self_alive` | 1 each | – (0 or 1) |
| `self_hurt_time` | 1 | 10 ticks |
| `target_present` | 1 | – |
| `target_rel_pos`, `target_rel_vel` | 3, 3 | 8 blocks, 0.5 blocks per tick |
| `target_distance` | 1 | 8 blocks |
| `target_yaw`, `target_pitch` | 1, 1 | 180, 90 |
| `target_bearing` | 2 | 180, 90 (the turn that faces the opponent) |
| `target_sneaking`, `target_sprinting`, `target_using_item`, `target_swinging`, `target_on_fire` | 1 each | – |
| `target_hurt_time` | 1 | 10 |
| `target_main_hand`, `target_off_hand`, `target_armor_item` | 1, 1, 4 | – |
| `target_health`, `target_absorption` | 1, 1 | 20 (see below) |

and derived ones:

| Feature | Values | What |
|---|---|---|
| `self_vel_local` | 3 | own velocity as (forward, right, up) of the agent: forward is where it faces (its yaw; pitch not used). Divided by 0.5 |
| `target_rel_vel_local`, `target_rel_pos_local` | 3, 3 | the opponent's relative velocity and position in that same frame |
| `self_yaw_sincos` | 2 | sine and cosine of the yaw: no jump at +-180 |
| `target_bearing_sincos` | 3 | sine and cosine of the bearing's yaw part, and its pitch part / 90 |

- Nothing is clipped: a knockback can push a velocity past 1. Item indices and the world
  position have no scale; leave them out or encode them yourself. `normalize=False` gives raw
  values.
- **`target_health` and `target_absorption`** are zero unless the run says the server shows its
  players each other's health (`target_health=True` in the `FleetConfig`; the env sets it when an
  observation asks for one of them). The fair-play rule and its switches:
  [design.md](design.md#what-the-agent-observes-one-place-gamerlfair_playrs).
- The `"sword"` preset (16 values) is the first training stage's view: `self_vel_local`,
  `self_yaw`, `self_pitch`, `self_health`, `self_attack_cooldown`, `self_on_ground`,
  `self_hurt_time`, `self_alive`, `target_present`, `target_rel_vel_local`, `target_distance`,
  `target_health`. Add your own to `observations.PRESETS`, or pass a list.
- The observation is fixed when the env is made (a policy's input size can't change under it). To
  change it, make another env; several envs with different observations and actions can share
  one running fleet (`HorizonVecEnv(fleet=env.fleet, observation=..., ...)`).

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
python/.venv/bin/python python/tests/e2e_shard.py        # model B: 4 image clients in 2 processes; kick, kill
python/.venv/bin/python python/tests/e2e_gym.py          # M4: the Gymnasium envs on 4 + 1 real clients
python/.venv/bin/python python/bench/throughput.py --ns 2,8,16,32   # the Gymnasium env: steps/s, CPU, RSS as N grows
python/.venv/bin/python python/bench/fleet.py --ns 2,8,16,32   # the fleet below it (--infer-ms 10)
python/.venv/bin/python python/bench/fleet.py --ns 8,32 --obs image --image 64x64   # the same with frames (+ GPU memory)
python/.venv/bin/python python/bench/fleet.py --ns 32 --obs image --per-process 8      # model B: 8 clients per process
python/.venv/bin/python python/bench/transport.py        # binary socket vs the JSON control socket
python3 python/bench/baseline.py . [--fleet 10]          # CPU and RSS per idle client
cargo test -p horizon-rl-proto                           # the wire format
cargo test -p horizon-client-render --lib gfx::luma      # the luma pass on the GPU (skips without one)
cargo test --profile fast -p horizon-client --lib -- agent::rl   # the fair-play view, kill parsing
```
