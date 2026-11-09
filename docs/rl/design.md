# Horizon RL: design

Status: **draft for review** (2026-11-09). Nothing here is built yet. The client baseline is measured
(see [Baseline](#baseline-measurements)); the transport numbers come with M1.

Goal: many headless 26.3 clients, each an offline-mode player on a remote (Paper) server, driven
from Python through a Gymnasium env for PvP self-play. Per-instance CPU and RAM are the main costs.
A fair-play filter makes sure each agent gets only what a human with a vanilla client could see.

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

**Plan:** build A first (M1, M2). It is simple and robust. Then measure RSS per instance and the
share that `Resources` and the GPU device take. Build B as `--clients K` only if sharing saves a lot
(I expect it to matter mostly for image mode, because of the device). The supervisor and the Python
side treat "instance" and "process" separately from the start, so B needs no protocol change.

## Transport

Each client gets one **Unix domain socket** for control and one **shared-memory region** (a file in
`/dev/shm`, mmapped by both sides) for observations.

- Shared memory holds a header (`magic`, `version`, `tick`, `seq`, flags), the fixed-layout `f32`
  state vector, the `u8` grayscale frame (double-buffered), and a fixed-size info block. Python
  reads them as zero-copy numpy views.
- The socket carries small binary messages (little-endian, fixed size). `HELLO` sends the config,
  `STEP{seq, action, frame_skip}`, `RESET`, `CLOSE`. Each reply is `READY{seq, tick}` after the
  client writes the observation. A dead socket means the instance died. That gives us fault
  detection for free.
- Why not the others:
  - **PyO3:** it would put the game in the Python process (or need a broker), which loses
    process isolation and adds the GIL.
  - **ZeroMQ:** it adds a native dependency on both sides and gives nothing over UDS for local
    one-to-one links.
  - **The JSON socket:** parsing and formatting the full entity list costs too much per step.
- **To measure** (M1, `python/bench/transport.py`): round-trip time and CPU per step for (1)
  `horizon-ctl state` JSON and (2) binary UDS + shm, at 1/10/30 instances. I expect a few µs for
  UDS against roughly 1 ms for JSON. Either is small next to a 50 ms tick, but the CPU cost × N is
  what matters. The doc gets the real numbers.

## Timing

The server runs in real time at 20 TPS. The client keeps its own 20 Hz tick, and Python never
blocks it.

- `step(action)` sends the action. The client applies it at the **start of its next tick**, holds
  it for `frame_skip` ticks, and replies once the observation after the last of those ticks has
  been written.
- The action is a held state, as with a human's keys. Movement, jump, sprint, sneak and use stay
  pressed until the next action changes them. Attack is a click: a press on the first tick and a
  release on the next. The camera delta is spread over the `frame_skip` ticks (or applied on the
  first tick, configurable).
- **Stalls show.** If Python is late, the client keeps ticking with the last action (the
  configurable alternative is to release all input). `info` reports `ticks_elapsed` (ticks since
  the previous observation), `late_ticks` (ticks run without a fresh action), and the client's own
  `tick_overrun_ms` (its tick took over 50 ms). Python can then drop those transitions or flag them.
- **Lockstep later:** `HELLO` carries `mode = realtime | lockstep`. Only `realtime` exists now. In
  lockstep, the client waits for each `STEP` before ticking. The protocol stays the same.

## Actions

The actions mirror player input. Each one goes through the same functions that window events
reach (`keys.press/release` with `Source::Agent`, `mouse_mapping`, `mouse.on_move`), so the server
sees the normal movement, swing, interact and use packets, with nothing extra.

| Field | Type |
|---|---|
| `forward`, `back`, `left`, `right`, `jump`, `sprint`, `sneak` | bool (held) |
| `attack` | bool (one click per step, press then release) |
| `use` | bool (held: shield, bow, food) |
| `hotbar` | 0 = no change, 1..9 = select |
| `camera` | `[dyaw, dpitch]` degrees: continuous `Box`, or discrete bins (Python-side wrapper; bins configurable) |

We convert camera degrees to mouse counts with `mouse::motion_for_turn` and the player's
sensitivity. The game turns from those counts as it would from a real mouse, so pitch is clamped as
in vanilla. Optional human caps (off by default): the largest turn per tick, and clicks per second.

The agent cannot reach `goto`, `look-at`, aimed `attack ID`, commands, chat sending or teleports.
The RL socket has no message for any of them.

## Observations

`--obs state|image|both`. In `state` mode the client never opens a renderer.

### Fair-play filter (one place)

`game/rl/fair_play.rs` is the only code that reads `Game` for observations. It builds a narrow
`PlayerView` from what the player perceives, and the encoder only sees that view. The rules are in
its module doc:

- **Self** (HUD and F3 about yourself): position, velocity (from position deltas), yaw and pitch,
  health, absorption, food level, armor bar value, XP level and progress, attack cooldown, selected
  slot, the hotbar and the off hand (item index + count), worn armor (item index), on-ground,
  sprinting, sneaking, using-item progress, own hurt flash (`hurt_time`), own effects (HUD icons).
  Saturation is out: vanilla doesn't show it.
- **Other players**: only if visible this tick, which means all of the following:
  1. in a loaded chunk within render distance;
  2. inside the view frustum (the player's FOV and the observation's aspect ratio);
  3. line of sight: a ray from the eye to at least one of several points on the hitbox (head,
     center, feet) is not blocked by a view-blocking block (glass and other see-through blocks let
     it pass);
  4. not a spectator, and not invisible. An invisible player wearing armor or holding an item
     shows only those parts, as vanilla draws them. To start with, we leave such players out.
- **Fields per visible player**: relative position, relative velocity (from this agent's own
  sightings, never from network velocity), relative head yaw and pitch, pose flags (sneaking,
  swimming or gliding, using item / blocking, swinging), hurt flash (`hurt_time > 0`), on fire,
  main hand, off hand and four armor items as item indices.
- **Never present**: another player's health, absorption, food, armor points or effects; entities
  that fail the visibility test; network entity IDs. Slots are sorted by distance each tick. A slot
  may carry a short-lived track number that starts over whenever that player leaves view.
- **Up to `max_players` slots** (default 4). Each has a `present` flag, and an empty slot is zeros.
- **Optional block grid** (`--block-grid R`, off by default): block categories (air, solid, liquid,
  climbable, damaging) in a (2R+1)³ cube around the feet, with R ≤ 2. These blocks touch the player
  and a human feels them, so this does not count as seeing through walls. Open question: limit the
  grid to air-exposed blocks?
- **Sound events** (allowed, phase 2): the last few sounds heard, as category and direction.

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

| `info` key | Source (what a player's client sees) |
|---|---|
| `health`, `health_delta`, `damage_taken` | `ClientboundSetHealth` |
| `hit_landed` | our attack swung at a player in reach, and that player's hurt animation or damage event arrived within a few ticks |
| `kills`, `deaths`, `killer_was_visible` | death messages: system chat with `death.attack.*` translation keys whose arguments name us (parsed from the component, not English text) |
| `died` | `ClientboundPlayerCombatKill` for ourselves |
| `respawned` | `ClientboundRespawn`. After death the client presses Respawn automatically (the button a human clicks) |
| `disconnected`, `reason`, `reconnects` | the connection and the reconnect loop |
| `ticks_elapsed`, `late_ticks`, `tick_overrun_ms` | timing (above) |

Episodes: `terminated = True` when the agent dies. The next `reset()` returns the observation after
respawn. A kick, disconnect or crash gives `truncated = True` with `info["reset_reason"]`, and the
supervisor reconnects or restarts the instance with backoff.

Server-side options we could add later, since they need hidden information and stay out of the
client: the opponent's exact health and damage dealt, kill credit when the death message is
ambiguous, and match results from a Paper plugin over a side channel.

## Fault isolation and reconnects

- The supervisor (Python `horizon_rl.launch`, or a `horizon-rl` CLI on top of it) owns every
  process. Each one gets its own scratch `HORIZON_HOME`, `--player-dir`, socket and shm file.
- When a socket or process dies, the supervisor restarts it with exponential backoff and the env
  reports a reset for that slot. Other instances are unaffected.
- When the client is kicked or disconnected, it reconnects itself (with backoff) instead of
  showing the Disconnected screen. The supervisor watches only for hangs: no tick for 5 s means a
  kill and a restart.
- Joins are staggered (default 250 ms). Paper's `bukkit.yml` `settings.connection-throttle`
  (default 4000 ms per IP) would otherwise reject N clients from one IP. Set it to `-1` on the
  training server.

## Python package

`python/horizon_rl/` (installable with `pip install -e python/`):

- `HorizonClient`: one instance (socket, shm, numpy views).
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
   telemetry and its server, cosmetics, friends and relay, heartbeat, official, the mod registry,
   port-pack, bench. Then CI workflows and docs that only cover them. **I'll ask before this step.**
3. Delete the in-game extras behind `horizon-extras` (cosmetics and emotes, Horizon settings pages,
   Discord, friends screens) and their harnesses, only where a harness covers only removed code.
4. Then work driven by profiling ([profiling.md](../profiling.md)): the streaming worker count
   (1 per instance), render distance 2 (the client's view distance in `ClientInformation`, so the
   server sends fewer chunks), skipping particles, sky and HUD where they feed no observation, and
   the size of `Resources`.

What stays: protocol, movement, combat, inventory, entity tracking, the render path, parity
harnesses for those, the dedicated server (it's the stand-in).

## Where the code goes

- `crates/horizon-client/src/game/rl/`: `mod.rs` (session, socket, tick hook), `fair_play.rs`
  (the filter), `obs.rs` (encoding), `action.rs`, `signals.rs` (info), behind a Cargo feature `rl`.
- `crates/horizon-rl-proto/`: the wire and shm layout, with no dependencies, so the format is in
  one place and its tests run quickly.
- `python/horizon_rl/`, `docs/rl/README.md`.

## Baseline measurements

Measured 2026-11-09 on the target desktop (16 threads, 23 GB RAM, Linux 7.3, Mesa), `--profile fast`, at the
branch point `7eb057384c`. One local `horizon-client --dedicated` (offline, peaceful, view distance 10). Each
client joins with `--server 127.0.0.1:PORT --name …` and fresh `HORIZON_HOME`/XDG/game folders, then stands still
at spawn. After a 15 s settle, it is sampled for 40 s from `/proc` (CPU = utime+stime of all threads, % of one core).
Script: `python/bench/baseline.py` (lands with M1).

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
- Still to measure: the transport (JSON ctl vs binary UDS+shm, M1), and CPU under motion and combat (M2).
