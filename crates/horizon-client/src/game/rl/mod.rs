//! The RL session (docs/rl/design.md): Python drives this client through a Unix socket and reads
//! observations from shared memory, in the wire format of `horizon-rl-proto`.
//!
//! - `--rl-socket PATH` listens for one Python connection at a time; `--rl-shm PATH` is the
//!   observation file (normally in `/dev/shm`); `--rl-target NAME` names the opponent (default:
//!   the nearest other player).
//! - Timing: a `STEP` received before a tick is applied at that tick's start and held for its
//!   frame skip; after the last of those ticks the observation is written and `READY` sent.
//!   Ticks that run while Python still owes the next step keep the held keys and count as late.
//! - Input goes through the paths window events take (`keys`, `mouse_mapping`), so the server
//!   sees a player's normal packets. The camera is absolute: the step's yaw and pitch.
//! - Only [`fair_play`] reads the game for the observation. The reward signals (`info`) are kept
//!   here, apart from it.
//! - The client stays on its server: off it (a kick, a lost connection, a failed join) it joins
//!   again after a backoff ([`reconnect`]). Steps are still answered there, one per 50 ms, with
//!   `connected` 0, so Python never blocks on a client that has no world to tick.
//! - Every frame bumps the shared memory's heartbeat (`BEAT_OFF`) for the supervisor.
//! - Headless, a client whose next step is owed waits a little for it before its tick
//!   ([`Game::rl_await_step`]), and its tick clock moves by the wait. Clients started at different
//!   times thus come to tick just after Python sends their steps, so a batch of N clients doesn't
//!   wait on the one whose tick came last. The waits are bounded ([`STEP_GRACE`], [`SHIFT_PER_TICK`]):
//!   a slow Python slows a client's ticks by 5 % at most and still shows as late ticks.

mod fair_play;
mod image;
mod reconnect;
mod shard;

pub(in super::super) use image::parse_image_size;
pub(in super::super) use shard::{Peer, ended as shard_ended};

use super::super::{Game, Screen};
use fair_play::{SelfView, TargetView};
use horizon_client::{held::Source, multiplayer, title::Mode};
use horizon_protocol::registry::Builtin;
use horizon_rl_proto::{self as proto, INFO, INFO_LEN, STATE_LEN, ShmLayout, button, kind};
use std::{
    fs::File,
    io::{ErrorKind, Read, Write},
    os::unix::{
        fs::FileExt,
        net::{UnixListener, UnixStream},
    },
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{event::MouseButton, keyboard::KeyCode};

/// The command line's RL options (`--rl-socket` turns the session on).
#[derive(Clone, Debug, Default)]
pub(in super::super) struct RlOptions {
    pub socket: PathBuf,
    pub shm: Option<PathBuf>,
    pub target: Option<String>,
    /// `--rl-step-grace MS`: the longest wait for an owed step ([`STEP_GRACE`] when unset).
    pub step_grace_ms: Option<u64>,
    /// `--rl-image WxH`: grayscale frames of the view in the shared memory ([`image`]).
    pub image: Option<[u32; 2]>,
    /// `--rl-hud`: those frames show the HUD (hidden by default, as F1).
    pub hud: bool,
    /// `--rl-client`: more clients in this process ([`shard`]).
    pub peers: Vec<Peer>,
}

/// The held keys, by their action bit, as the default key bindings name them.
const KEYS: [(u32, KeyCode); 7] = [
    (button::FORWARD, KeyCode::KeyW),
    (button::BACK, KeyCode::KeyS),
    (button::LEFT, KeyCode::KeyA),
    (button::RIGHT, KeyCode::KeyD),
    (button::JUMP, KeyCode::Space),
    (button::SPRINT, KeyCode::ControlLeft),
    (button::SNEAK, KeyCode::ShiftLeft),
];

/// A hit counts as ours when the target's damage arrives this many ticks after our swing at most
/// (the server sends it the tick it handles the attack; this covers latency).
const HIT_WINDOW_TICKS: u64 = 10;

/// The longest a client waits at one tick for an owed step.
const STEP_GRACE: Duration = Duration::from_millis(20);
/// The waiting each tick earns (2.5 ms of 50: ticks run at least 95 % as often as 20 Hz), and
/// the most that can be saved up.
const SHIFT_PER_TICK: Duration = Duration::from_micros(2500);
const SHIFT_SAVED: Duration = Duration::from_millis(40);

/// What happened since the last observation (reset when one is written).
#[derive(Default, Debug, Clone, PartialEq)]
struct Signals {
    ticks: u32,
    late: u32,
    overrun_ms: f32,
    damage_taken: f32,
    was_hit: u32,
    target_damage: f32,
    target_hurt: u32,
    hit_landed: u32,
    attacks: u32,
    died: u32,
    respawned: u32,
    disconnected: u32,
}

struct Conn {
    stream: UnixStream,
    decoder: proto::Decoder,
    hello: bool,
}

pub(super) struct Session {
    options: RlOptions,
    listener: UnixListener,
    conn: Option<Conn>,
    shm: File,
    shm_path: String,
    layout: ShmLayout,
    items: Option<Arc<Builtin>>,
    frame_skip: u32,
    /// A step received, applied at the next tick's start.
    pending: Option<proto::Step>,
    /// The step being held: its seq and the ticks left.
    running: Option<(u64, u32)>,
    /// `READY` was sent and the next step hasn't come yet.
    owed: bool,
    reset: bool,
    held: u32,
    signals: Signals,
    kills: u32,
    deaths: u32,
    /// The target's entity id, kept while it is tracked.
    target: Option<u32>,
    target_seen: Option<[f64; 3]>,
    target_health: Option<f32>,
    own_health: Option<f32>,
    alive: bool,
    last_attack: Option<u64>,
    ticks: u64,
    /// `Chat.received` when the chat was last read.
    chat_seen: u64,
    /// The heartbeat (frames run).
    beat: u64,
    /// When the session last ticked itself, off its world.
    idle_tick: Instant,
    backoff: reconnect::Backoff,
    was_connected: bool,
    disconnects: u32,
    reconnects: u32,
    /// The waiting for steps still allowed (see [`STEP_GRACE`]).
    shift_budget: Duration,
    /// Image observations, with `--rl-image`.
    frames: Option<image::Frames>,
}

impl Session {
    pub(super) fn open(options: RlOptions) -> std::io::Result<Self> {
        let _ = std::fs::remove_file(&options.socket);
        let listener = UnixListener::bind(&options.socket)?;
        listener.set_nonblocking(true)?;
        let shm_path = options.shm.clone().unwrap_or_else(|| {
            let mut p = options.socket.clone().into_os_string();
            p.push(".shm");
            p.into()
        });
        let [w, h] = options.image.unwrap_or([0, 0]);
        let layout = ShmLayout::new(w, h);
        let shm = File::options()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&shm_path)?;
        shm.set_len(layout.size as u64)?;
        Ok(Session {
            listener,
            conn: None,
            shm,
            shm_path: shm_path.to_string_lossy().into_owned(),
            layout,
            items: None,
            frame_skip: 1,
            pending: None,
            running: None,
            owed: false,
            reset: false,
            held: 0,
            signals: Signals::default(),
            kills: 0,
            deaths: 0,
            target: None,
            target_seen: None,
            target_health: None,
            own_health: None,
            alive: false,
            last_attack: None,
            ticks: 0,
            chat_seen: 0,
            beat: 0,
            idle_tick: Instant::now(),
            backoff: reconnect::Backoff::default(),
            was_connected: false,
            disconnects: 0,
            reconnects: 0,
            shift_budget: SHIFT_SAVED,
            frames: options.image.map(image::Frames::new),
            options,
        })
    }

    fn send(&mut self, kind: u32, payload: &[u8]) {
        if let Some(c) = self.conn.as_mut() {
            // The socket is non-blocking; a small reply always fits its buffer. If Python stopped
            // reading, the connection is dead to us.
            c.stream.set_nonblocking(false).ok();
            let ok = c.stream.write_all(&proto::frame(kind, payload)).is_ok();
            c.stream.set_nonblocking(true).ok();
            if !ok {
                self.conn = None;
            }
        }
    }

    /// The item registry id + 1 of `name`, 0 for none or unknown.
    fn item_id(&self, name: Option<&str>) -> u32 {
        let (Some(items), Some(name)) = (self.items.as_ref(), name) else {
            return 0;
        };
        items
            .get("minecraft:item")
            .and_then(|ids| ids.id(name))
            .map_or(0, |id| id as u32 + 1)
    }

    fn info(&self, game_connected: bool) -> [f32; INFO_LEN] {
        let s = &self.signals;
        let mut out = [0.0f32; INFO_LEN];
        let mut put = |name: &str, v: f32| out[proto::offset(INFO, name)] = v;
        put("tick", self.ticks as f32);
        put("ticks_elapsed", s.ticks as f32);
        put("late_ticks", s.late as f32);
        put("tick_overrun_ms", s.overrun_ms);
        put("connected", f32::from(u8::from(game_connected)));
        put("alive", f32::from(u8::from(self.alive)));
        put("health", self.own_health.unwrap_or(0.0));
        put("damage_taken", s.damage_taken);
        put("was_hit", s.was_hit as f32);
        put("target_present", f32::from(u8::from(self.target.is_some())));
        put("target_health", self.target_health.unwrap_or(f32::NAN));
        put("target_damage", s.target_damage);
        put("target_hurt", s.target_hurt as f32);
        put("hit_landed", s.hit_landed as f32);
        put("attacks", s.attacks as f32);
        put("died", s.died as f32);
        put("respawned", s.respawned as f32);
        put("kills_total", self.kills as f32);
        put("deaths_total", self.deaths as f32);
        put("disconnected", s.disconnected as f32);
        put("disconnects_total", self.disconnects as f32);
        put("reconnects_total", self.reconnects as f32);
        let frames = self.frames.as_ref();
        put(
            "frame_fresh",
            frames.map_or(0.0, |f| f32::from(u8::from(f.fresh))),
        );
        put("frame_ms", frames.map_or(0.0, |f| f.ms));
        out
    }
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The names of the item registry by id, for the welcome.
fn item_names(items: &Builtin) -> Vec<&str> {
    items
        .get("minecraft:item")
        .map(|ids| ids.names.iter().map(String::as_str).collect())
        .unwrap_or_default()
}

/// A kill of ours in a death message: `death.*` translated with the victim first and the killer
/// second (`CombatTracker.getDeathMessage`), the killer being `me` and the victim someone else.
pub(super) fn kill_by(component: &serde_json::Value, me: &str) -> bool {
    use horizon_client::scoreboard::comp_text;
    let Some(key) = component.get("translate").and_then(|k| k.as_str()) else {
        return false;
    };
    let Some(args) = component.get("with").and_then(|w| w.as_array()) else {
        return false;
    };
    if !key.starts_with("death.") || args.len() < 2 {
        return false;
    }
    let name = |v: &serde_json::Value| match v.as_str() {
        Some(s) => s.to_owned(),
        None => comp_text(v).plain(),
    };
    name(&args[1]) == me && name(&args[0]) != me
}

impl Game<'_> {
    /// Opens the session at start when `--rl-socket` was given.
    pub(in super::super) fn rl_start(&mut self) {
        let Some(options) = self.options.rl.clone() else {
            return;
        };
        // The No AI switch turns off every surface an agent drives the game through.
        if self.settings.extras.no_ai {
            log::warn!("RL session not opened: {}", horizon_core::no_ai::REFUSAL);
            return;
        }
        match Session::open(options) {
            Ok(mut s) => {
                s.items = super::super::remote::builtin_registries(&self.options.resources);
                log::info!("RL session on {}", s.options.socket.display());
                // The frames show the HUD only when asked (F1's hideGui otherwise).
                if s.options.image.is_some() && !s.options.hud {
                    self.ui.hide_gui = true;
                }
                self.agent.rl = Some(Box::new(s));
            }
            Err(e) => log::error!("RL session: {e}"),
        }
    }

    fn rl_alive(&self) -> bool {
        self.player
            .as_ref()
            .is_some_and(|p| p.health.health() > 0.0)
            && self.screen != Screen::Dead
            && self.title.is_none()
    }

    /// Every frame: the socket (connections, messages), the death screen's Respawn button, and a
    /// reset waiting for the player to be alive.
    pub(in super::super) fn rl_frame(&mut self) {
        let Some(mut s) = self.agent.rl.take() else {
            return;
        };
        // An observation still waiting for its frame: no frame was drawn since its tick (no
        // renderer, or the frame failed). It goes out with the last frame.
        if let Some(seq) = s.frames.as_mut().and_then(|f| f.due.take()) {
            if let Some(f) = s.frames.as_mut() {
                f.fresh = false;
            }
            self.rl_write(&mut s, seq);
        }
        match s.listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(true).ok();
                // A new connection replaces the old one: its held input is let go.
                s.conn = Some(Conn {
                    stream,
                    decoder: proto::Decoder::default(),
                    hello: false,
                });
                s.pending = None;
                s.running = None;
                s.owed = false;
                s.reset = false;
                self.rl_release(&mut s);
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {}
            Err(e) => log::warn!("RL accept: {e}"),
        }
        let mut buf = [0u8; 4096];
        let mut closed = false;
        let mut messages = Vec::new();
        if let Some(c) = s.conn.as_mut() {
            loop {
                match c.stream.read(&mut buf) {
                    Ok(0) => {
                        closed = true;
                        break;
                    }
                    Ok(n) => c.decoder.push(&buf[..n]),
                    Err(e) if e.kind() == ErrorKind::WouldBlock => break,
                    Err(e) if e.kind() == ErrorKind::Interrupted => {}
                    Err(_) => {
                        closed = true;
                        break;
                    }
                }
            }
            loop {
                match c.decoder.next_message() {
                    Ok(Some(m)) => messages.push(m),
                    Ok(None) => break,
                    Err(_) => {
                        closed = true;
                        break;
                    }
                }
            }
        }
        let mut reset_release = false;
        for (k, payload) in messages {
            let hello = s.conn.as_ref().is_some_and(|c| c.hello);
            match k {
                kind::HELLO => match proto::Hello::decode(&payload) {
                    Some(h) if h.version == proto::VERSION => {
                        s.frame_skip = h.frame_skip.max(1);
                        let names = s.items.as_deref().map(item_names).unwrap_or_default();
                        let welcome = proto::welcome(
                            &s.shm_path,
                            &s.layout,
                            &names,
                            &self.options.player_name(),
                        );
                        if let Some(c) = s.conn.as_mut() {
                            c.hello = true;
                        }
                        s.send(kind::WELCOME, welcome.as_bytes());
                    }
                    _ => {
                        let why = format!("this client speaks version {}", proto::VERSION);
                        s.send(kind::ERROR, why.as_bytes());
                        closed = true;
                    }
                },
                kind::STEP if hello => match proto::Step::decode(&payload) {
                    Some(step) => s.pending = Some(step),
                    None => closed = true,
                },
                kind::RESET if hello => {
                    s.pending = None;
                    s.running = None;
                    s.reset = true;
                    // An episode starts with nothing held: the keys of the last one (often
                    // still pressed at death) let go, so the ticks before the new episode's
                    // first step are spent standing.
                    reset_release = true;
                }
                _ => closed = true,
            }
        }
        if reset_release {
            self.rl_release(&mut s);
        }
        if closed {
            s.conn = None;
            s.pending = None;
            s.running = None;
            s.owed = false;
            s.reset = false;
            self.rl_release(&mut s);
        }
        // DeathScreen's Respawn button, which is active 20 ticks after death (`delayTicker`).
        if self.screen == Screen::Dead
            && self.ui.death_time >= 20
            && let Err(e) = self.respawn_player()
        {
            log::warn!("RL respawn: {e}");
        }
        self.rl_stay_connected(&mut s);
        // Off the world no client tick runs; the session ticks itself at the same 20 Hz so steps
        // are still answered.
        let now = Instant::now();
        if self.title.is_none() && !self.pause {
            s.idle_tick = now;
        } else if now.saturating_duration_since(s.idle_tick) >= Duration::from_millis(50) {
            s.idle_tick = now;
            self.agent.rl = Some(s);
            self.rl_tick_pre();
            self.rl_tick_post(Duration::ZERO);
            s = self.agent.rl.take().expect("the session is back");
        }
        s.beat += 1;
        if let Err(e) = s
            .shm
            .write_all_at(&s.beat.to_le_bytes(), proto::BEAT_OFF as u64)
        {
            log::error!("RL shm: {e}");
        }
        self.agent.rl = Some(s);
    }

    /// Before a headless frame: when Python owes the next step, wait for it (bounded, see
    /// [`STEP_GRACE`]). Returns the time waited, which the caller takes out of the tick clock so
    /// the tick moves rather than catching up.
    pub(in super::super) fn rl_await_step(&mut self) -> Duration {
        let in_world = self.title.is_none() && !self.pause;
        let Some(s) = self.agent.rl.as_mut() else {
            return Duration::ZERO;
        };
        let grace = s
            .options
            .step_grace_ms
            .map_or(STEP_GRACE, Duration::from_millis);
        let limit = s.shift_budget.min(grace);
        let Some(c) = s.conn.as_mut() else {
            return Duration::ZERO;
        };
        // During a reset Python waits for READY, it owes no step.
        if !in_world
            || !s.owed
            || s.reset
            || s.pending.is_some()
            || limit.is_zero()
            || c.decoder.has_message()
        {
            return Duration::ZERO;
        }
        let start = Instant::now();
        let mut buf = [0u8; 4096];
        c.stream.set_nonblocking(false).ok();
        loop {
            let left = limit.saturating_sub(start.elapsed());
            if left.is_zero() {
                break;
            }
            c.stream.set_read_timeout(Some(left)).ok();
            match c.stream.read(&mut buf) {
                Ok(0) => break, // closed: the frame's read finds it
                Ok(n) => {
                    c.decoder.push(&buf[..n]);
                    if c.decoder.has_message() {
                        break;
                    }
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        c.stream.set_read_timeout(None).ok();
        c.stream.set_nonblocking(true).ok();
        let waited = start.elapsed();
        s.shift_budget = s.shift_budget.saturating_sub(waited);
        waited
    }

    /// Notices a lost connection, and joins the `--server` again when the backoff allows.
    fn rl_stay_connected(&mut self, s: &mut Session) {
        let connected = self.remote.is_some();
        let now = Instant::now();
        let off = match self.title.as_ref().map(|t| &t.mode) {
            Some(Mode::Multiplayer(multiplayer::Screen::Disconnected { reason, .. })) => {
                Some(reason.clone())
            }
            Some(Mode::Error { message, .. }) => Some(message.clone()),
            _ => None,
        };
        if s.was_connected && !connected {
            s.disconnects += 1;
            s.signals.disconnected += 1;
            let reason = off.clone().unwrap_or_default();
            log::info!("RL: disconnected: {reason}");
            s.send(
                kind::NOTE,
                format!(
                    "{{\"event\":\"disconnected\",\"reason\":{}}}",
                    serde_json::Value::from(reason)
                )
                .as_bytes(),
            );
        }
        s.was_connected = connected;
        if connected {
            s.backoff.connected(now);
            return;
        }
        let Some(address) = self.options.server.clone() else {
            return;
        };
        if off.is_none() || !s.backoff.off(now) {
            return;
        }
        s.reconnects += 1;
        let attempt = s.backoff.attempts();
        log::info!("RL: joining {address} again (attempt {attempt})");
        s.send(
            kind::NOTE,
            format!("{{\"event\":\"reconnecting\",\"attempt\":{attempt}}}").as_bytes(),
        );
        let name = self.options.player_name();
        if let Err(e) = self.connect_server(&address, &name) {
            log::warn!("RL reconnect: {e}");
        }
    }

    /// Lets go of every key and button the session holds.
    fn rl_release(&mut self, s: &mut Session) {
        for (bit, key) in KEYS {
            if s.held & bit != 0 {
                self.keys.release(&key, Source::Agent);
            }
        }
        if s.held & button::USE != 0 {
            let _ = self.mouse_mapping(MouseButton::Right, false, Source::Agent);
        }
        s.held = 0;
    }

    fn rl_apply(&mut self, s: &mut Session, a: &proto::Action) {
        let want = a.buttons & button::ALL;
        for (bit, key) in KEYS {
            match (want & bit != 0, s.held & bit != 0) {
                (true, false) => {
                    self.keys.press(key, Source::Agent);
                }
                (false, true) => {
                    self.keys.release(&key, Source::Agent);
                }
                _ => {}
            }
        }
        match (want & button::USE != 0, s.held & button::USE != 0) {
            (true, false) => {
                let _ = self.mouse_mapping(MouseButton::Right, true, Source::Agent);
            }
            (false, true) => {
                let _ = self.mouse_mapping(MouseButton::Right, false, Source::Agent);
            }
            _ => {}
        }
        s.held = want & !button::ATTACK;
        if (1..=9).contains(&a.hotbar) {
            self.selected = a.hotbar as usize - 1;
        }
        if !a.yaw.is_nan() {
            self.yaw = a.yaw;
        }
        if !a.pitch.is_nan() {
            self.pitch = a.pitch.clamp(-90.0, 90.0);
        }
        if want & button::ATTACK != 0 {
            // A click: the press is the attack (Minecraft.startAttack), the release follows.
            let _ = self.mouse_mapping(MouseButton::Left, true, Source::Agent);
            let _ = self.mouse_mapping(MouseButton::Left, false, Source::Agent);
            s.signals.attacks += 1;
            s.last_attack = Some(s.ticks);
        }
    }

    /// One client tick (`Game::tick`) between the session's two hooks.
    pub(in super::super) fn rl_tick(&mut self) -> super::super::Result<()> {
        self.rl_tick_pre();
        let started = std::time::Instant::now();
        horizon_client::spans::time("tick", || self.tick())?;
        self.rl_tick_post(started.elapsed());
        Ok(())
    }

    /// Before each client tick: start a step that arrived.
    fn rl_tick_pre(&mut self) {
        let Some(mut s) = self.agent.rl.take() else {
            return;
        };
        if let Some(step) = s.pending.take() {
            let frame_skip = if step.frame_skip == 0 {
                s.frame_skip
            } else {
                step.frame_skip
            };
            self.rl_apply(&mut s, &step.action);
            s.running = Some((step.seq, frame_skip));
            s.owed = false;
        } else if s.owed {
            s.signals.late += 1;
        }
        self.agent.rl = Some(s);
    }

    /// After each client tick: the tick's signals, and the observation when a step is done.
    fn rl_tick_post(&mut self, took: Duration) {
        let Some(mut s) = self.agent.rl.take() else {
            return;
        };
        s.ticks += 1;
        s.signals.ticks += 1;
        s.shift_budget = (s.shift_budget + SHIFT_PER_TICK).min(SHIFT_SAVED);
        let over = took.as_secs_f32() * 1000.0 - 50.0;
        if over > 0.0 {
            s.signals.overrun_ms += over;
        }
        let alive = self.rl_alive();
        if alive && !s.alive && s.ticks > 1 && s.own_health.is_some() {
            s.signals.respawned += 1;
        }
        // Death: the player was alive and its health reached 0 (the death screen opens).
        if s.alive && !alive && self.title.is_none() {
            s.signals.died += 1;
            s.deaths += 1;
        }
        self.rl_scan_chat(&mut s);
        s.alive = alive;
        let health = self.player.as_ref().map(|p| p.health.health());
        if let (Some(before), Some(now)) = (s.own_health, health)
            && now < before
        {
            s.signals.damage_taken += before - now;
        }
        s.own_health = health;
        self.rl_track_target(&mut s);
        // A reset ends on a tick once the player is alive, so the next step starts on the next.
        if s.reset && alive {
            s.reset = false;
            s.signals = Signals::default();
            s.signals.ticks = 1;
            self.rl_observe(&mut s, 0);
        } else if let Some((seq, left)) = s.running {
            if left <= 1 {
                s.running = None;
                self.rl_observe(&mut s, seq);
            } else {
                s.running = Some((seq, left - 1));
            }
        }
        self.agent.rl = Some(s);
    }

    /// Keeps the target: the named player, else the nearest other player, while it is tracked.
    fn rl_track_target(&mut self, s: &mut Session) {
        let mobs = &self.mobs;
        let tracked = |id: u32| {
            mobs.iter()
                .find(|m| m.id == id && !m.removed && &*m.kind.id == "minecraft:player")
        };
        if s.target.is_some_and(|id| tracked(id).is_none()) {
            s.target = None;
            s.target_seen = None;
            s.target_health = None;
        }
        if s.target.is_none() {
            let me = self.player.as_ref().map(|p| p.position);
            let named = s.options.target.as_ref().and_then(|want| {
                self.remote_player_names()
                    .into_iter()
                    .find(|(_, name)| name == want)
                    .map(|(id, _)| id)
            });
            s.target = match (&s.options.target, named) {
                (Some(_), found) => found.filter(|id| tracked(*id).is_some()),
                (None, _) => me.and_then(|me| {
                    self.mobs
                        .iter()
                        .filter(|m| !m.removed && &*m.kind.id == "minecraft:player")
                        .min_by(|a, b| {
                            let d = |m: &&horizon_client::mobs::Mob| {
                                (m.position.x - me.x).powi(2) + (m.position.z - me.z).powi(2)
                            };
                            d(a).total_cmp(&d(b))
                        })
                        .map(|m| m.id)
                }),
            };
        }
        let Some(m) = s.target.and_then(tracked) else {
            return;
        };
        let health = m.health;
        if let Some(before) = s.target_health
            && health < before
        {
            s.signals.target_damage += before - health;
        }
        s.target_health = Some(health);
    }

    /// A step's observation is due: written now, or after the frame drawn for it.
    fn rl_observe(&mut self, s: &mut Session, seq: u64) {
        match s.frames.as_mut() {
            Some(f) => f.due = Some(seq),
            None => self.rl_write(s, seq),
        }
    }

    /// Writes the observation and info, then tells Python (`READY`).
    fn rl_write(&mut self, s: &mut Session, seq: u64) {
        let me = self.rl_self_view(s);
        let target = s
            .target
            .and_then(|id| self.mobs.iter().find(|m| m.id == id))
            .map(|m| {
                let previous = s.target_seen;
                TargetView::of(m, previous, |name| s.item_id(name))
            });
        s.target_seen = target.as_ref().map(|t| t.pos);
        let mut state = [0.0f32; STATE_LEN];
        fair_play::encode(&me, target.as_ref(), &mut state);
        let info = s.info(self.remote.is_some());
        let layout = s.layout;
        let ok = s
            .shm
            .write_all_at(&floats(&state), layout.state_off as u64)
            .and_then(|_| s.shm.write_all_at(&floats(&info), layout.info_off as u64))
            .and_then(|_| s.shm.write_all_at(&layout.header(seq, s.ticks), 0));
        if let Err(e) = ok {
            log::error!("RL shm: {e}");
        }
        s.signals = Signals::default();
        s.owed = true;
        s.send(kind::READY, &proto::Ready { seq, tick: s.ticks }.encode());
    }

    fn rl_self_view(&mut self, s: &Session) -> SelfView {
        let alive = self.rl_alive();
        // The HUD's attack indicator (`Gui.extractCrosshair`: getAttackStrengthScale(0)).
        let cooldown = self.hand_strength.scale(0.0, self.main_hand_attack_speed());
        let using = self.using.is_some();
        let sneaking = self.keys.held(horizon_client::actions::Action::Sneak);
        let stack = |st: &Option<horizon_client::items::ItemStack>| {
            st.as_ref()
                .map_or((0, 0), |st| (s.item_id(Some(&st.item)), st.count))
        };
        let hotbar: [(u32, u32); 9] =
            std::array::from_fn(|i| self.inventory.get(i).map_or((0, 0), stack));
        let offhand = stack(&self.offhand);
        let armor_items = std::array::from_fn(|i| stack(&self.armor[i]).0);
        let (selected, yaw, pitch) = (self.selected, self.yaw, self.pitch);
        let Some(p) = self.player.as_mut() else {
            return SelfView::default();
        };
        let armor = p
            .attribute("minecraft:armor")
            .or_else(|_| p.attribute("armor"))
            .unwrap_or(0.0) as i32;
        SelfView {
            alive,
            pos: [p.position.x, p.position.y, p.position.z],
            vel: [p.velocity.x, p.velocity.y, p.velocity.z],
            yaw,
            pitch,
            health: p.health.health(),
            absorption: p.health.absorption(),
            food: p.health.food.food,
            armor,
            attack_cooldown: cooldown,
            selected,
            hotbar,
            offhand,
            armor_items,
            on_ground: p.ground.on_ground,
            sprinting: p.sprinting,
            sneaking,
            using_item: using,
            hurt_time: p.health.hurt_animation_time,
        }
    }

    /// The server reported entity `id` hurt (`ClientboundDamageEventPacket`, through
    /// `pvp_hurt`): `own` when it is this player, `by_me` whether we were the damage's direct
    /// entity, when the packet names one.
    pub(in super::super) fn rl_note_damage(&mut self, id: i32, own: bool, by_me: Option<bool>) {
        let Some(s) = self.agent.rl.as_mut() else {
            return;
        };
        if own {
            s.signals.was_hit += 1;
        } else if s.target.is_some_and(|t| i64::from(t) == i64::from(id)) {
            s.signals.target_hurt += 1;
            let recent_swing = s
                .last_attack
                .is_some_and(|at| s.ticks.saturating_sub(at) <= HIT_WINDOW_TICKS);
            if by_me.unwrap_or(recent_swing) {
                s.signals.hit_landed += 1;
            }
        }
    }

    /// Our kills, from the death messages that reached the chat since the last tick.
    fn rl_scan_chat(&mut self, s: &mut Session) {
        let received = self.ui.chat.received;
        let new = received.saturating_sub(s.chat_seen).min(64) as usize;
        s.chat_seen = received;
        if new == 0 {
            return;
        }
        let me = self.own_profile_name();
        s.kills += self
            .ui
            .chat
            .recent_components(new)
            .iter()
            .filter(|c| kill_by(c, &me))
            .count() as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_death_message_naming_us_second_is_our_kill() {
        let kill = json!({"translate": "death.attack.player", "with": ["Bot_2", "Bot_1"]});
        assert!(kill_by(&kill, "Bot_1"));
        assert!(!kill_by(&kill, "Bot_2"), "the victim didn't kill");
        let fell = json!({"translate": "death.fell.accident.generic", "with": ["Bot_2"]});
        assert!(!kill_by(&fell, "Bot_1"));
        let chat = json!({"translate": "chat.type.text", "with": ["Bot_1", "Bot_1"]});
        assert!(!kill_by(&chat, "Bot_1"), "only death messages count");
    }

    #[test]
    fn names_in_components_are_read_as_text() {
        let kill = json!({
            "translate": "death.attack.player.item",
            "with": [{"text": "Bot_2", "insertion": "Bot_2"}, {"text": "Bot_1"}, {"text": "[Sword]"}]
        });
        assert!(kill_by(&kill, "Bot_1"));
    }
}
