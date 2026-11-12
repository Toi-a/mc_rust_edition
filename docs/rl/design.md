# Horizon RL: design

Status: **reviewed** (2026-11-10). M1 (one client from Python), M2 (a supervised fleet,
[measured](#m2-measurements)) and M3 (grayscale image observations, [measured](#m3-measurements))
are built, process model B as an option ([measured](#model-b-measurements)), and the Gymnasium
envs (M4, [measured](#m4-measurements)); removal (M5) is next.

Goal: many headless 26.3 clients, each an offline-mode player on a remote (Paper) server, driven
from Python through a Gymnasium env for PvP self-play. Per-instance CPU and RAM are the main costs.
One filter decides what the policy sees: itself and its one opponent, never the opponent's health (that goes to `info` for rewards).

## What the repo already gives us

| Piece | Where | Use for RL |
|---|---|---|
| `--headless`: no display, window or renderer; own 50 ms loop | `game/headless.rs` | The state-only instance. It already skips meshing (`world_update.rs`) and uses silent audio. |
| `--offscreen`: renders into a texture | `gfx.rs` (`Renderer.offscreen`), `window_handler.rs` | The image path. It **still opens a hidden winit window**, so it needs a display server. |
| `--server`, `--name`, `--player-dir`, `--game-dir`, `--home`, `--render-distance` | `game/launch_options.rs` | Join at start with an offline name, one folder per player. |
| Agent control socket, `horizon-ctl` | `control.rs`, `game/control.rs`, `game/agent.rs` | Shows where input gets injected: `keys.press(_, Source::Agent)`, `mouse_mapping`, `mouse.on_move`. Kept for debugging. Not the RL interface: it uses JSON, has `goto`/`look-at`/aimed `attack`, and `entities` returns health and every tracked entity. |
| Splitscreen: several `Game`s in one process, one wgpu device, shared `Resources` | `game/split.rs` | Shows that `start_game(&resources, …)` can run several clients over one copy of the tables. |
| The client-side entity list | `Game.mobs` (`mobs.rs::Mob`: `health`, `hurt_time`, pose, equipment…) | The raw input to the fair-play filter. It holds **more than a player sees**: synced health, entities behind walls. |

## Process model

There are two candidates. We measure both before choosing (M1/M2).

**A. One process per client.** The supervisor starts N `horizon-client --headless --rl …`
processes. Isolation comes free: a crash or kick stays in that process. The cost is one copy of
`Resources` (block facts, items, recipes, models, lang) per process. With images, it is also one
wgpu device per process, and a GPU driver context costs 100–300 MB of RSS on Mesa/NVIDIA.

**B. Shards: one process runs K clients**, the way splitscreen does: K `Game`s over one
`Resources` and one wgpu device, each ticked on a small thread pool. A shard crash restarts its K
clients, and each one shows up in Python as a reset. Risks: process-wide state assumes one client.
There are 26 `thread_local!`/`OnceLock` in `game/`, for example the split `OPTIONS`, the cache dir
and the narrator. Each one has to be checked.

**Decided:** A, one process per client (M1, M2), is the default: simple and robust. B is built
too (after M3), behind a switch: `--rl-client name=N,socket=S[,shm=M][,target=T][,dir=D]`
(repeatable) adds clients to a headless process, and `FleetConfig.clients_per_process` (K, default
1) makes the fleet use it ([measured](#model-b-measurements)). No protocol change: each client
still has its own socket and shm file.

How B is built (`game/rl/shard.rs`):

- The process's own client loads everything as before. The others are `Game`s made the way
  splitscreen makes its players' (`start_game` over the same `Resources`, leaked for the process's
  lifetime, and a clone of the packs), each on **its own thread** running the unchanged headless
  loop. `Game` isn't `Send`, so each is built on its thread; what it is built from is (`Resources`
  is `Sync`; packs, UI resources, the renderer and audio are `Send`).
- With images, their renderers are made on the main thread over the first one's device, queue and
  pipelines (`Renderer::new`'s `previous`, as a resource reload does), then moved to their thread.
  One device, one driver context; each client keeps its own textures, targets and meshes.
- Thread-locals are per client for free (one thread each); splitscreen's per-thread
  `options.txt` path gives each its own (`dir=`). The ~180 statics in the client crates were
  checked: read-only tables, mod hooks, process-wide settings and counters, the same for every
  client of one command line.
- **They live and die together.** When one client's loop ends (an error, a stop signal), the
  others end theirs and the process exits. The supervisor treats the K slots as one unit: a fault
  in any (crash, hang, timeout) restarts all K, each a truncated episode (`reset_reason` of the
  others: the same with " (shard)"). A kick stays one client's own: it rejoins by itself.

## Transport

Each client gets one **Unix domain socket** for control and one **shared-memory region** (a file in
`/dev/shm`, mmapped by both sides) for observations.

- Shared memory holds a 64-byte header (`HZRL`, version, `seq`, `tick`, the layout's offsets),
  the `f32` state vector, the `f32` info vector, and (M3) the `u8` frame. Python maps it once and
  reads numpy views. The game writes it before `READY` and not again until the next step ends, so
  no lock is needed.
- The socket carries `[kind u32][len u32][payload]` messages, little-endian. `HELLO{version,
  frame_skip}` gets `WELCOME`: JSON with the shm path, the state and info fields as `[name,
  offset, len]`, the item registry and the player's name, so Python never hard-codes the layout.
  Then `STEP{seq, frame_skip, buttons, hotbar, yaw, pitch}` (28 bytes), `RESET`, `CLOSE`; each
  step or reset is answered by `READY{seq, tick}`. A dead socket means the instance died.
  The format lives in one crate, `crates/horizon-rl-proto`.
- Why not the others:
  - **PyO3:** it would put the game in the Python process (or need a broker), which loses
    process isolation and adds the GIL.
  - **ZeroMQ:** it adds a native dependency on both sides and gives nothing over UDS for local
    one-to-one links.
  - **The JSON control socket:** measured below. It answers once per frame, so the two requests
    an observation needs (`state`, `entities`) take two frames: it can't keep up with 20 Hz.

**Measured** (2026-11-10, `python/bench/transport.py --seconds 30`; one headless client and a second
player on the local server; one observation per tick):

| | Binary UDS + shm | JSON control socket (`state` + `entities 32`) |
|---|---|---|
| Observations/s | **20.0** (every tick) | 10.0 (one request per frame) |
| Python CPU per observation | **78 µs** | 229 µs |
| Game CPU (idle: 0.87 %) | **0.87 %** (no measurable cost) | 1.03 % |
| Bytes per observation | 400 | 2 116 |
| Wait from request to answer (median) | 50 ms (the tick) | 100 ms (two frames) |

## Timing

The server runs in real time at 20 TPS. The client keeps its own 20 Hz tick, and Python never
blocks it.

- `step(action)` sends the action. The client applies it at the **start of its next tick**, holds
  it for `frame_skip` ticks (default **1**: a new action every tick), and replies once the
  observation after the last of those ticks has been written.
- **Sticky keys.** The action is a held state, as with a human's keys. Movement, jump, sprint, sneak
  and use stay pressed until a later action changes them. Attack is a click: a press and a release
  at the start of the step's first tick (the press is what attacks, as `Minecraft.startAttack`).
- **Stalls show.** If Python is late, the client keeps ticking with the last action (sticky keys).
  `info` reports `ticks_elapsed` (ticks since the previous observation), `late_ticks` (ticks run
  without a fresh action), and the client's own `tick_overrun_ms` (its tick took over 50 ms). Python
  can then drop those transitions or flag them.
- **Ticks line up after the steps** (headless, built in M2). A batch of N clients waits for the
  last of them, but each client ticks on its own phase: once the policy takes time between the
  last `READY` and the next sends, every client whose tick falls in that gap ticks without its
  action. So a headless client whose step is owed waits for it a little before its tick, and its
  tick clock moves by the wait (`--rl-step-grace`, 20 ms by default, 0 turns it off). The clients
  soon all tick just after Python sends. The waits are bounded: each tick earns 2.5 ms of waiting
  (at most 40 ms saved), so a Python that is slow for good slows a client by 5 % at most and still
  shows as late ticks. No wait happens off the world or during a reset. Measured with 32 fighting
  clients and 10 ms of inference per step: 2 275 late ticks in 25 s without it, 15 with it.
- The headless loop wakes when the tick clock's next tick is due (not on a fixed 50 ms cadence),
  so each frame carries one tick, never zero then two.
- **Lockstep later:** `HELLO` carries `mode = realtime | lockstep`. Only `realtime` exists now. In
  lockstep, the client waits for each `STEP` before ticking. The protocol stays the same.

## Actions

Keys and clicks go through the same functions that window events reach (`keys.press/release` with
`Source::Agent`, `mouse_mapping`), so the server sees the normal movement, swing, interact and use
packets, with nothing extra.

| Field | Type |
|---|---|
| `forward`, `back`, `left`, `right`, `jump`, `sprint`, `sneak` | bool (held) |
| `attack` | bool (one click per step, press then release) |
| `use` | bool (held: shield, bow, food) |
| `hotbar` | 0 = no change, 1..9 = select |
| `look` | `[yaw, pitch]` in degrees, **absolute**: the direction to face from this tick on |

**Camera: absolute angles.** The agent says where to look each tick; there is no mouse simulation.
The client sets the player's yaw and pitch (pitch clamped to ±90 as in vanilla) before the tick's
movement, so the rotation reaches the server in the normal movement packet, as a mouse turn would.
A `NaN` component keeps the current angle. Python wrappers can offer deltas or bins on top of it.
Human caps, applied in Python: the largest turn per tick (default: 90 degrees of yaw, any pitch)
and, optionally, clicks per second.

The agent cannot reach `goto`, `look-at`, aimed `attack ID`, commands, chat sending or teleports.
The RL socket has no message for any of them.

## Observations

`--obs state|image|both`. In `state` mode the client never opens a renderer.

### What the agent observes (one place: `game/rl/fair_play.rs`)

The setting is a 1v1 duel on flat ground: each agent has exactly one opponent, its **target**, and
no other player is around. So there is no visibility test: the target is observed whenever the
client tracks it, in front, behind or out of view. `fair_play.rs` is still the only code that reads
`Game` for observations, and it still decides what the policy may see.

- **Target**: `--target NAME` names it; without it, the client takes the nearest other player and
  keeps it until that player leaves (then it picks again). `info["target"]` names it.
- **Self** (HUD and F3 about yourself): position, velocity, yaw and pitch, health, absorption, food
  level, armor bar value, attack cooldown, selected slot, the hotbar and the off hand (item index +
  count), worn armor (item index), on-ground, sprinting, sneaking, using-item progress, own hurt
  flash (`hurt_time`), own effects.
- **Target fields**: `present`, relative position, relative velocity, its yaw and pitch (absolute
  and relative to ours), pose flags (sneaking, using item / blocking, swinging), hurt flash, on
  fire, main hand, off hand and four armor items as item indices.
- **Kept out of the observation**: the target's health, absorption, food, armor points and effects.
  The target's health goes to `info` for rewards (below), not to the policy. One flag
  (`--obs-target-health`, off) would add it if you decide otherwise.
- No block grid: fights are on flat ground.

Encoding: one flat `f32` vector with a documented layout, which the client returns in `HELLO` as
`[(name, offset, len)]` so the Python `Box` and a `Dict` view stay in sync with the Rust side.
Item indices come from the 26.3 item registry order, which the client also returns.

### Images

Built in M3 (`game/rl/image.rs`, `gfx/luma.rs`): `--rl-image WxH` (W a multiple of 4, 8..512).

- **No display**: `--headless --render` (which `--rl-image` turns on) builds the game's normal
  `Renderer` with no window: `Renderer::new` takes `Option<Arc<Window>>` and, without one, opens
  the device with no surface and draws into its offscreen target at W x H. The headless loop
  stays as it was.
- **Once per observation**: the observation of a step is written after the frame drawn right
  after its last tick, so frame and state show the same tick (`Game::rl_render_frame`, which
  `Game::render` calls). Between observations the headless game draws nothing: with frame skip 4,
  one frame per 4 ticks.
- **Luma on the GPU**: the offscreen picture (8-bit, after the tone curve and the HUD, as the
  display would show it) is copied into a buffer, a compute pass packs Rec. 601 luma bytes, and
  they come back through a mapped staging buffer straight into the shm frame slot. No PNG, no CPU
  colour work. HDR stays off (docs/hdr-passes.md rule 8: a capture takes the SDR picture). The
  read waits for the GPU; at 64x64 the whole frame (draw + read) costs ~1–3 ms.
- **Nothing extra drawn**: the HUD is hidden unless `--rl-hud` (vanilla's F1, which also hides
  the hand and all name tags, so no separate `--no-nametags` is needed).
- **Cheap settings**: the fleet writes `options.txt` (`RENDER_OPTIONS`): no clouds, minimal
  particles, no smooth lighting, biome blend, mipmaps or entity shadows; render distance 2.
- **State-only pays nothing**: without `--rl-image` no renderer or GPU device is opened.
- `info` gains `frame_fresh` (this observation's frame was drawn for it; 0 keeps the last one,
  e.g. if the GPU failed) and `frame_ms`. Python: `Session.frame()` / `Fleet.frames` (n, H, W).
- **BAR1**: wgpu's allocator puts mappable upload buffers in the device-local, host-visible
  heap. On NVIDIA without Resizable BAR that is the 256 MB BAR1 window, shared by every process:
  ~20 MB per client, so from ~12 image clients on new ones fail to open the GPU ("Parent device
  is lost", `NVRM: can't alloc VA space` in the kernel log). `HORIZON_NO_BAR_UPLOADS=1` (in our
  vendored `third_party/wgpu-hal`, set by the fleet) keeps them in system memory: 0.75 MB of BAR1
  per client, and no measurable cost (N = 8: 9.9 vs 9.4 % CPU, 3.0 vs 2.9 ms per frame).

## Rewards, episodes and `info`

These are signals only. Python builds the reward.

| `info` key | Source |
|---|---|
| `health`, `health_delta`, `damage_taken` | own health (`ClientboundSetHealth`) |
| `target_health`, `target_health_delta` | the target's synced health (entity data the server sends to tracking clients). `NaN` when the server hides it (some plugins spoof it) |
| `was_hit` | we took damage this step (own damage event / hurt flash start) |
| `hit_landed` | the target took damage this step after our attack: its damage event or hurt flash start, or its health dropping, within a few ticks of our swing |
| `kills`, `deaths` | death messages: system chat with `death.attack.*` translation keys whose arguments name us (parsed from the component, not English text) |
| `died` | `ClientboundPlayerCombatKill` for ourselves |
| `respawned` | `ClientboundRespawn`. After death the client presses Respawn automatically (the button a human clicks) |
| `connected`, `disconnected`, `disconnects_total`, `reconnects_total` | the connection and the reconnect loop; the reason comes in a `NOTE` |
| `ticks_elapsed`, `late_ticks`, `tick_overrun_ms` | timing (above) |

Episodes: `terminated = True` when the agent dies. The next `reset()` returns the observation after
respawn. A kick, disconnect or crash gives `truncated = True` with `info["reset_reason"]`, and the
supervisor reconnects or restarts the instance with backoff.

Server-side options for later: damage dealt as the server computed it, and match results from a
Paper plugin over a side channel.

## Fault isolation and reconnects

Built in M2. The supervisor is `horizon_rl.fleet.Fleet` (Python), started from a `FleetConfig`
or the `horizon-rl` command.

- It owns every process. Each one gets its own scratch homes and game folder, socket and shm
  file, and `HORIZON_JOB_THREADS`/`HORIZON_WORKER_THREADS` = `workers` (default 1).
- **The client stays on its server** (`game/rl/reconnect.rs`). Off it (the Disconnected or error
  screen: a kick, a timeout, the server closing, a failed join), it joins the `--server` again
  after 1, 2, 4, 8, 16, then every 30 s; 30 s connected starts the delays over. Unlike the
  auto-reconnect extra for people, kicks are retried too. The game sends a `NOTE`
  (`{"event":"disconnected","reason":…}`), counts `disconnected`/`disconnects_total`/
  `reconnects_total` in `info`, and keeps answering steps at 20 Hz off the world (`connected` 0),
  so Python never blocks on it.
- **Heartbeat**: the game bumps a `u64` in the shm header (`BEAT_OFF` = 56) every frame. The
  supervisor kills a process whose heartbeat stands still for `hang_timeout` (5 s), that exits,
  that closes its socket, or that doesn't answer a step within `step_timeout` (10 s), and starts it
  again after 1, 2, 4 … 30 s (60 s healthy starts them over).
- **What Python sees**: the step that meets a fault is `truncated` with `info["reset_reason"]`
  (`crash (…)`, `hang`, `timeout`, `disconnected: <reason>`). A death is `terminated`. An ended slot
  resets by itself: it is left out of the next steps (`active` False) until it is alive again, and
  its first observation comes with `info["episode_start"]`. A fault between episodes is named in
  that first info's `reset_reason`. A `RESET` lets go of every held key, so the tick or two
  before the new episode's first step (its observation arrives between batches) are spent
  standing, reported in `late_ticks`. This is Gymnasium's next-step autoreset, which M4 maps onto. No fault touches another slot (`python/tests/e2e_fleet.py`
  checks a kill, a kick, a SIGSTOP hang and a death on four fighting clients).
- Joins are staggered (`stagger`, default 250 ms; restarts too). Paper's `bukkit.yml`
  `settings.connection-throttle` (default 4000 ms per IP) would otherwise reject N clients from
  one IP. Set it to `-1` on the training server.
- Headless clients don't open the controller backend (gilrs and its udev thread). With one
  worker, a client runs 6 threads (25 before).

## Python package

`python/horizon_rl/` (installable with `pip install -e python/`):

- `HorizonClient`: one instance (socket, shm, numpy views), blocking.
- `Fleet` / `FleetConfig` (M2): N instances, stepped together without threads (one `selectors`
  loop), supervised; `horizon-rl` runs one from the command line or a TOML file.
- `HorizonEnv(gymnasium.Env)`: one agent (a fleet of one, so it is supervised too).
- `HorizonVecEnv(gymnasium.vector.VectorEnv)` (M4, `env.py`): sends all actions, then waits for
  all observations. Instances run in parallel in real time, so this costs no extra threads.
  - Spaces: the observation space comes from the `WELCOME` layout and `obs`; the action space
    from `actions.py`'s `ActionCodec`: buttons `MultiBinary(9)`, hotbar `Discrete(10)`, and the
    camera as absolute angles, deltas (continuous) or bins (discrete), chosen by `camera=`. Deltas
    and bins are added in Python to the last observed yaw and pitch; the wire stays absolute.
    They are a setting of the env, not Gymnasium wrappers: the turn needs the env's last
    observation, and one codec serves the single, vector and PettingZoo envs.
  - The action space is built from parts (decided 2026-11-12, for training in stages): which
    buttons the policy has, whether it has the hotbar, the camera mode, and `flat=True` for one
    `MultiDiscrete` or `Box` instead of a `Dict` (Stable-Baselines3 takes no `Dict` actions).
    `camera="aim"` faces the opponent for the policy from the observed bearing: aim assist, a
    training aid for a first stage and the one deliberate exception to "no aim helpers".
  - Human caps, in the same codec. Decided 2026-11-12: the policy acts every tick (frame skip
    1), and a tick turns at most 90 degrees of yaw and any pitch (`max_turn=(90, 0)`, the
    default). `max_cps` stays off by default.
  - `release()`: one step that holds nothing and keeps the view, for the pauses of a trainer (a
    PPO update takes seconds; the game runs on with sticky keys). `sb3.py` has SB3's `VecEnv`
    over the env and the callback that releases after each rollout.
  - **Autoreset**: Gymnasium's next-step mode, stretched over the reset's real duration. The
    fleet's `active` mask and `episode_start` go into `infos`, with `valid = active and not
    episode_start`: the rows that are real transitions. An inactive agent's row repeats its last
    observation (so it stays inside the space) with reward 0. A strict next-step autoreset would
    block every agent while one respawns or restarts, and the others' ticks would run without
    actions.
  - Rewards: none built in (0); `reward=` takes a function of the infos (`rewards.duel`).
- `HorizonParallelEnv` (PettingZoo `ParallelEnv`, `parallel.py`): an optional wrapper over the
  vector env. The server does the matchmaking, so agents don't interact through the env.
- `examples/random_agent.py` and `bench/throughput.py` (steps/s per instance and in total, CPU %
  and RSS per instance as N grows, max N that holds 20 Hz).

## Server compatibility

The client speaks only 26.3, protocol 777. The server must accept that: either Paper built for
26.3, or Paper on another version with ViaVersion (plus ViaBackwards if that version is newer). Set
`online-mode=false` and `connection-throttle: -1`. No older protocols get added. Locally, the
repo's own dedicated server (`horizon-client --dedicated`, offline mode) stands in for Paper.

## Removal plan (M5, step by step, each step builds and passes tests)

1. **No deletions yet.** Build with `--no-default-features` plus a new `rl` feature. That drops
   audio, controllers, Discord, friends and heartbeat. Measure.
2. Delete the `launcher/` workspace and the Horizon-layer crates the RL build doesn't link:
   telemetry and its server, cosmetics, friends and relay, heartbeat, the mod registry, port-pack,
   bench. Then CI workflows and docs that only cover them. Decided (2026-11-10): these go, one crate
   group per commit, each building and passing tests.
3. Delete the in-game extras behind `horizon-extras` (cosmetics and emotes, Horizon settings pages,
   Discord, friends screens) and their harnesses, only where a harness covers only removed code.
4. Then work driven by profiling ([profiling.md](../profiling.md)): the streaming worker count
   (1 per instance), render distance 2 (the client's view distance in `ClientInformation`, so the
   server sends fewer chunks), skipping particles, sky and HUD where they feed no observation, and
   the size of `Resources`.

What stays: protocol, movement, combat, inventory, entity tracking, the render path, parity
harnesses for those, the dedicated server (it's the stand-in).

## Where the code goes

- `crates/horizon-client/src/game/rl/` (the module `game::agent::rl`, kept beside the other agent
  surfaces): `mod.rs` (session, socket, actions, info signals) and `fair_play.rs` (the filter and
  the state encoding). No Cargo feature: without `--rl-socket` the hooks are one `None` check. The
  hooks stay out of the big files (`game.rs` only calls `rl_tick`; damage comes through
  `pvp_hurt`, deaths from the alive→dead change, kills from the chat's new death messages).
- `game/rl/image.rs` (frames, M3) and `game/rl/shard.rs` (process model B).
- `crates/horizon-rl-proto/`: the wire and shm layout, with no dependencies, so the format is in
  one place and its tests run quickly.
- `python/horizon_rl/`, `docs/rl/README.md`.

## Baseline measurements

Measured 2026-11-09 on the target desktop (16 threads, 23 GB RAM, Linux 7.3, Mesa), `--profile fast`, at the
branch point `7eb057384c`. One local `horizon-client --dedicated` (offline, peaceful, view distance 10). Each
client joins with `--server 127.0.0.1:PORT --name …` and fresh `HORIZON_HOME`/XDG/game folders, then stands still
at spawn. After a 15 s settle, it is sampled for 40 s from `/proc` (CPU = utime+stime of all threads, % of one core).
Script: `python/bench/baseline.py`.

| Client | CPU % (1 core) | RSS MB | PSS MB | Threads | Join s |
|---|---|---|---|---|---|
| `--headless --render-distance 2` | 1.1 | 386 | 358 | 25 | 1.0 |
| `--headless --render-distance 10` | 1.4 | 407 | 379 | 25 | 1.0 |
| `--offscreen --render-distance 2 --width 128 --height 128` | 6.9 | 598 (max 616) | 514 | 34 | 2.5 |
| `--offscreen --render-distance 10` (1280×720) | 9.3 | 747 (max 766) | 667 | 34 | 1.0 |
| 10 × `--headless --render-distance 2` at once | 0.7 each, 7.0 total | 391 each | 343 each | – | all joined, all alive |

The server itself was at 7.9 % CPU and 508 MB with 10 players.

What this says:

- **Memory is the limit, not CPU.** An idle headless client costs ~1 % of a core but ~390 MB. At 23 GB that
  caps out around 45–50 headless instances (~30 with images) before any training process. Idle is the best
  case: PvP with movement, entities and chunk loading will raise CPU, and M2 measures that under scripted motion.
- **Most of the RSS is fixed per process.** Render distance 2→10 adds only ~20 MB, so the bulk is the loaded
  `Resources` (35 723 block states, 1 658 items, models, lang) plus the allocator and thread stacks. This makes
  the shard model (B) and trimming what headless loads (models, atlases, sounds) the main RAM levers. The
  offscreen device adds ~200 MB per process, as expected.
- **Threads to cut**: headless still starts 15 streaming workers and the controller backend (it found a
  gamepad). Both are candidates for M1/M5 defaults (`--workers 1`, no gamepad).
- Still to measure: CPU under motion and combat, and many instances at once (M2, below).

## M2 measurements

Measured 2026-11-12 on the same desktop, `--profile fast`, with `python/bench/fleet.py --ns
2,8,16,32,40 --seconds 30 --settle 10` (and `--infer-ms 10` for the last two rows). Each N gets a
fresh local dedicated server (offline, normal world, seed 1234, `spawn_radius 0`) and N clients in
pairs with diamond swords. The scripted fighter (`horizon_rl/scripted.py`, observation only) turns
to its partner, runs at it, strafes, jumps, swings at full cooldown, dies and respawns; one Python
process steps all N every tick (frame skip 1). 30 s are measured after 10 s of fighting. CPU is
utime + stime of all threads from `/proc`, in % of one core.

| N | inference | client CPU % (mean / max) | client RSS MB (mean / max) | PSS MB | vector steps/s | late ticks per 1000 steps | tick overruns | server CPU % | Python CPU % |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0 | 1.05 / 1.20 | 396 / 399 | 361 | 20.00 | 0 | 0 | 6 | 0.4 |
| 8 | 0 | 1.07 / 1.23 | 396 / 401 | 349 | 20.00 | 0 | 0 | 10 | 1.1 |
| 16 | 0 | 1.23 / 1.43 | 395 / 407 | 346 | 20.01 | 0 | 0 | 12 | 2.2 |
| 32 | 0 | 1.43 / 1.70 | 394 / 404 | 343 | 20.01 | 0 | 0 | 19 | 5.5 |
| 40 | 0 | 1.53 / 1.73 | 385 / 405 | 334 | 20.00 | 0 | 0 | 23 | 7.6 |
| 32 | 10 ms | 1.54 / 1.76 | 394 / 402 | 343 | 20.00 | 0 | 0 | 21 | 5.7 |
| 40 | 10 ms | 1.58 / 1.77 | 388 / 403 | 337 | 19.99 | 0 | 0 | 23 | 7.7 |

(Tick overruns: client ticks over 50 ms. Over 30 s, N = 40 played 685 hits and 93 deaths; no slot
was truncated or restarted.)

What this says:

- **20 Hz holds at every N measured, up to 40**, with no late ticks, also with 10 ms of
  inference per step. CPU is far from the limit: 40 fighting clients take 0.6 of a core in total
  (~1.5 % each), the 16-thread machine is ~95 % idle.
- **Memory sets the largest N.** ~390 MB RSS (~340 MB PSS) per client, flat in N. 40 clients used
  ~15.4 GB of the 23 GB. N = 40 is the most measured; by memory, ~48–50 fit with ~2 GB left for the
  trainer, without images. That's 4–5× the ~10 Java clients. More needs less memory per process
  (the shards of model B, or trimming what headless loads: M5).
- Starting is quick: 40 clients are in and alive 14 s after the command (250 ms stagger).
- Measurement pitfall: reading `/proc/PID/smaps_rollup` for every client during the run stalls
  the stepping loop enough to cause late ticks; the benchmark reads RSS from `statm` and PSS once,
  afterwards.
- The local stand-in server spends ~5 cores on a **flat** world as soon as players are on it, even
  idle (a normal world: ~6 %). That is a Horizon dedicated-server issue, not the clients'; the e2e
  tests still use flat, the benchmark a normal world. On Paper this doesn't apply.

## M3 measurements

Measured 2026-11-12 on the same desktop (RTX 2070 SUPER 8 GB, NVIDIA 620.70), `--profile fast`,
`python/bench/fleet.py --obs image --image 64x64` (same fights as M2, frame skip 1: a frame
every tick). GPU memory is `nvidia-smi`'s total, divided by N.

| N | obs | client CPU % | RSS MB | PSS MB | GPU MB | frame ms | fresh frames | steps/s | late ticks |
|---|---|---|---|---|---|---|---|---|---|
| 8 | state | 3.36 | 395 | 348 | 0 | – | – | 20.01 | 0 |
| 8 | image 64x64 | 9.9 | 544 | 379 | 182 | 3.0 | 100 % | 19.99 | 0.2 / 1000 |
| 32 | image 64x64 | 3.44 | 546 | 374 | 181 | 1.5 | 100 % | 20.00 | 0 |

N = 40 with images wasn't completed: the session's memory guard stopped the run (40 × ~550 MB RSS
on a 23 GB desktop).

What this says:

- **20 Hz holds with a frame every tick at N = 32**, every frame fresh, no late ticks.
- **Memory per image client**: +150 MB RSS but only +30 MB PSS over state-only (the NVIDIA
  driver's libraries are shared), plus **~180 MB of GPU memory**. The design expected +200 MB of
  RAM; it is the GPU memory that costs. On this 8 GB card that caps N at ~42 with images; by
  RAM (PSS), ~50.
- **CPU % depends on the clock.** With the `powersave` governor and a lightly loaded machine the
  cores run slow, so the same work shows as more % of a core: state-only at N = 8 was 1.07 % in
  the M2 table (measured while a stray server kept the cores busy) and 3.36 % here, with the
  machine otherwise idle. At N = 32 the fleet keeps the clocks up itself. Compare rows only within
  one run; for absolute numbers use the `performance` governor (or count cycles).
- **The shard model (B)** would share one GPU device (driver context, block textures, pipelines)
  and the driver's per-process memory among K clients: the 180 MB of GPU memory and ~30 MB PSS
  per client are what it could save. Built after M3: [Model B measurements](#model-b-measurements).

## Model B measurements

Measured 2026-11-12 on the same desktop, now on the `performance` power profile (intel_pstate:
EPP `performance`), `python/bench/fleet.py --per-process K`, 30 s after 10 s of settling, pairs
fighting, frame skip 1. Per-client CPU, RSS and PSS are each process's divided by K. GPU memory is
`nvidia-smi`'s total over N.

| N | obs | K | client CPU % | RSS MB | PSS MB | GPU MB | frame ms | fresh frames | steps/s | late ticks | started |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | image 64x64 | 1 | 2.57 | 550 | 390 | 182 | 1.10 | 100 % | 20.01 | 0 | 6.5 s |
| 16 | image 64x64 | 2 | 2.68 | 364 | 261 | 190 | 1.14 | 100 % | 20.00 | 0 | 4.7 s |
| 16 | image 64x64 | 4 | 2.50 | 273 | 212 | 175 | 1.01 | 100 % | 20.00 | 0 | 4.1 s |
| 16 | image 64x64 | 8 | 2.39 | 227 | 175 | 168 | 0.95 | 100 % | 20.00 | 0 | 4.3 s |
| 32 | image 64x64 | 8 | 2.96 | 233 | 183 | 168 | 1.26 | 100 % | 20.00 | 0 | 5.8 s |
| 16 | state | 1 | 1.00 | 397 | 347 | 0 | – | – | 20.01 | 0 | 5.7 s |
| 16 | state | 8 | 0.91 | 156 | 152 | 0 | – | – | 20.00 | 0 | 2.8 s |

What this says:

- **RAM per client more than halves.** PSS drops from 390 to 175 MB per image client at K = 8 and from
  347 to 152 MB state-only. What is shared is the per-process part: `Resources`, the code, the
  driver's libraries and context. By RAM, this desktop (23 GB) goes from ~50 to ~100 image clients.
- **GPU memory barely moves** (182 → 168 MB per client). Sharing the device saves only ~15 MB. The
  rest belongs to each renderer (its block and entity textures, buffers, targets), so GPU memory
  still caps image mode at ~45 clients on 8 GB. To go further, renderers on one device would have
  to share their read-only textures, or headless rendering would need to allocate less: M5
  profiling work.
- **CPU is the same or a little lower** (~5–10 %), and 20 Hz holds with every frame fresh and no
  late ticks.
- **The cost is isolation**: one fault restarts K clients. With K = 8 that is 8 truncated episodes
  per crash. The default stays K = 1; use K > 1 when memory is the limit.
- The numbers in this table and the M2/M3 tables come from different power settings (M2/M3 used
  `powersave`), so compare CPU % within this table only. State-only N = 16 here, 1.00 %, is close
  to M2's 1.07 % (whose clocks a stray server kept high).

## M4 measurements

Measured 2026-11-12 on the same desktop (`performance` power profile), through the Gymnasium
vector env: `python/bench/throughput.py --ns … --seconds 30` (10 s of settling first), agents in
pairs on a fresh local server (normal world, seed 1234), frame skip 1. The policy is the scripted
fighter played through the Gymnasium action space, except the last row (random samples of the
action space). Per-client CPU, RSS and PSS are a process's divided by K. "Python" is the trainer's
side: the env, the supervisor and the scripted policy, with no inference.

| N | obs | K | steps/s per instance | transitions/s in total | client CPU % | RSS MB | PSS MB | GPU MB | Python CPU % | Python ms per vector step | late ticks per 1000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | state | 1 | 20.06 | 37.9 | 0.80 | 392 | 357 | – | 0.8 | 0.38 | 0 |
| 8 | state | 1 | 20.01 | 147.6 | 0.92 | 393 | 347 | – | 1.8 | 0.90 | 0 |
| 16 | state | 1 | 20.00 | 293.9 | 1.08 | 394 | 345 | – | 3.3 | 1.64 | 0 |
| 32 | state | 1 | 20.00 | 588.4 | 1.25 | 383 | 332 | – | 7.1 | 3.53 | 0 |
| 32 | state | 8 | 20.00 | 585.8 | 1.13 | 157 | 152 | – | 6.4 | 3.22 | 0 |
| 64 | state | 8 | 20.00 | 1174.3 | 1.60 | 159 | 154 | – | 15.6 | 7.81 | 0 |
| 32 | image 64x64 | 8 | 20.00 | 589.4 | 2.98 | 231 | 177 | 168 | 7.3 | 3.63 | 0 |
| 16 | both 64x64 (random policy) | 1 | 20.00 | 320.0 | 3.05 | 549 | 390 | 181 | 3.5 | 1.74 | 0 |

No slot was truncated or restarted in any row.

What this says:

- **The Gymnasium layer costs nothing that shows**: every row holds 20 steps per second per
  instance with no late ticks, and the clients' CPU and memory are the fleet's own (M2, M3 and
  model B tables).
- **The largest N measured that holds 20 Hz is 64** (state, K = 8): 1 174 transitions per second,
  ~10 GB of RAM, one core for all the clients together. Without model B, 32 was measured (12.5 GB);
  N = 40 held in M2. More than 64 wasn't tried.
- **Transitions are a little under N x 20** in the fighting rows: an agent that died gives none for
  the second its respawn takes (`valid` False). With the random policy nobody dies: exactly 320.
- **Python's share grows linearly**: ~0.11 ms per agent per step (about half is the scripted
  policy, which a trainer replaces by one batched inference). At N = 64 that is 7.8 ms of the 50 ms
  tick, leaving ~40 ms for inference before ticks would run late.
