# Horizon RL: design

Status: **reviewed** (2026-11-10). M1 (one client from Python) and M2 (a supervised fleet,
[measured](#m2-measurements)) are built; images (M3), the Gymnasium env (M4) and removal (M5) are
next.

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

**Decided:** A, one process per client (M1, M2). It is simple and robust. Then measure RSS per instance and the
share that `Resources` and the GPU device take. Build B as `--clients K` only if sharing saves a lot
(I expect it to matter mostly for image mode, because of the device). The supervisor and the Python
side treat "instance" and "process" separately from the start, so B needs no protocol change.

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
Optional human caps (off by default): the largest turn per tick, and clicks per second.

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

- An offscreen render of the player's normal view at W×H (64–128). The HUD is optional, and nothing
  extra is drawn: no hitboxes, outlines or nametags through walls (vanilla hides nametags behind
  walls anyway, but `--no-nametags` turns them off entirely).
- A small GPU pass converts the picture to luma `u8` (it reads the tonemapped LDR picture, per
  [hdr-passes.md](../hdr-passes.md)). We copy it into a mapped staging buffer, then memcpy it into
  the shm frame slot: no PNG and no CPU colour conversion.
- Rendering has to work **without a display**. Today `--offscreen` needs winit. New: `--headless
  --render`, where the headless loop also builds a `Renderer` with no surface (wgpu allows that,
  `open_device_with(window = None, offscreen = true)`). We render once per observation, not per
  tick: with `frame_skip` 4 that is 5 frames/s.
- Rendering costs: smallest render distance, no clouds or particles in the picture by default
  (configurable), shadows and other post effects off.

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
  that first info's `reset_reason`. No fault touches another slot (`python/tests/e2e_fleet.py`
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
- `HorizonEnv(gymnasium.Env)`: one agent.
- `HorizonVecEnv(gymnasium.vector.VectorEnv)`: sends all actions, then waits for all
  observations. Instances run in parallel in real time, so this costs no extra threads.
- `HorizonParallelEnv` (PettingZoo `ParallelEnv`): an optional wrapper. The server does the
  matchmaking, so agents don't interact through the env.
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
