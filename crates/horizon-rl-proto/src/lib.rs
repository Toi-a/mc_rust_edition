//! The RL environment's wire format and shared-memory layout ([docs/rl/design.md]).
//!
//! This crate is the one definition of the format. The Python side (`python/horizon_rl`) does not
//! hard-code offsets: it reads the layout from the [`welcome`] reply the game sends after `HELLO`.
//!
//! **Socket** (Unix domain, the game listens): every message is `[kind: u32][len: u32][payload]`,
//! little-endian. Python sends `HELLO`, then `STEP`/`RESET` and waits for `READY` after each;
//! `CLOSE` ends the session. Setup replies (`WELCOME`, `ERROR`) carry UTF-8 JSON; per-step
//! messages are fixed-size binary.
//!
//! **Shared memory** (a file, normally in `/dev/shm`): a [`HEADER_LEN`]-byte header, then the
//! state vector (`f32`s, laid out by [`STATE`]), then the info vector ([`INFO`]). The game writes
//! them before it sends `READY`, and doesn't write them again until the next `STEP` or `RESET`
//! completes, so Python may read them in place without locking.
//!
//! [docs/rl/design.md]: ../../../docs/rl/design.md

/// Bumped on any change to messages or layouts.
pub const VERSION: u32 = 1;
/// The first four bytes of the shared-memory file.
pub const SHM_MAGIC: [u8; 4] = *b"HZRL";
/// Bytes of the shared-memory header.
pub const HEADER_LEN: usize = 64;
/// The largest payload a message may have (the welcome JSON with the item registry is ~60 KB).
pub const MAX_PAYLOAD: usize = 1 << 20;

/// Message kinds.
pub mod kind {
    /// Python → game: [`super::Hello`].
    pub const HELLO: u32 = 1;
    /// Python → game: [`super::Step`].
    pub const STEP: u32 = 2;
    /// Python → game: no payload. `READY` comes once the player is alive in the world.
    pub const RESET: u32 = 3;
    /// Python → game: no payload. The game closes the connection.
    pub const CLOSE: u32 = 4;
    /// Game → Python: the session's layout as JSON ([`super::welcome`]).
    pub const WELCOME: u32 = 0x81;
    /// Game → Python: [`super::Ready`].
    pub const READY: u32 = 0x82;
    /// Game → Python: a UTF-8 reason, after which the game closes the connection.
    pub const ERROR: u32 = 0xff;
}

/// Bits of [`Action::buttons`]: held keys and clicks.
pub mod button {
    pub const FORWARD: u32 = 1 << 0;
    pub const BACK: u32 = 1 << 1;
    pub const LEFT: u32 = 1 << 2;
    pub const RIGHT: u32 = 1 << 3;
    pub const JUMP: u32 = 1 << 4;
    pub const SPRINT: u32 = 1 << 5;
    pub const SNEAK: u32 = 1 << 6;
    /// A left click (press and release) on the step's first tick.
    pub const ATTACK: u32 = 1 << 7;
    /// The right button, held.
    pub const USE: u32 = 1 << 8;
    pub const ALL: u32 = (1 << 9) - 1;
}

/// `HELLO`: what Python wants from the session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hello {
    pub version: u32,
    /// Ticks each `STEP` holds its action for when the step doesn't say (at least 1).
    pub frame_skip: u32,
}

/// One tick's input. Keys are held (sticky) until a later action changes them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Action {
    /// [`button`] bits.
    pub buttons: u32,
    /// 0 keeps the slot, 1..=9 selects that hotbar slot.
    pub hotbar: u32,
    /// Absolute yaw in degrees (Minecraft's: 0 = south, 90 = west); NaN keeps the current one.
    pub yaw: f32,
    /// Absolute pitch in degrees, clamped to -90..=90 (down is positive); NaN keeps it.
    pub pitch: f32,
}

/// `STEP`: apply `action` from the next tick for `frame_skip` ticks, then reply `READY`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Step {
    pub seq: u64,
    /// 0 uses the session's frame skip.
    pub frame_skip: u32,
    pub action: Action,
}

/// `READY`: the observation for step `seq` is in shared memory, written after client tick `tick`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ready {
    pub seq: u64,
    pub tick: u64,
}

pub const HELLO_LEN: usize = 8;
pub const STEP_LEN: usize = 28;
pub const READY_LEN: usize = 16;

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn f32_at(b: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

impl Hello {
    pub fn decode(b: &[u8]) -> Option<Self> {
        (b.len() == HELLO_LEN).then(|| Hello {
            version: u32_at(b, 0),
            frame_skip: u32_at(b, 4),
        })
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(HELLO_LEN);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&self.frame_skip.to_le_bytes());
        out
    }
}

impl Step {
    pub fn decode(b: &[u8]) -> Option<Self> {
        (b.len() == STEP_LEN).then(|| Step {
            seq: u64_at(b, 0),
            frame_skip: u32_at(b, 8),
            action: Action {
                buttons: u32_at(b, 12),
                hotbar: u32_at(b, 16),
                yaw: f32_at(b, 20),
                pitch: f32_at(b, 24),
            },
        })
    }
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(STEP_LEN);
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&self.frame_skip.to_le_bytes());
        out.extend_from_slice(&self.action.buttons.to_le_bytes());
        out.extend_from_slice(&self.action.hotbar.to_le_bytes());
        out.extend_from_slice(&self.action.yaw.to_le_bytes());
        out.extend_from_slice(&self.action.pitch.to_le_bytes());
        out
    }
}

impl Ready {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(READY_LEN);
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&self.tick.to_le_bytes());
        out
    }
    pub fn decode(b: &[u8]) -> Option<Self> {
        (b.len() == READY_LEN).then(|| Ready {
            seq: u64_at(b, 0),
            tick: u64_at(b, 8),
        })
    }
}

/// A framed message: kind, then the payload's length, then the payload.
pub fn frame(kind: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    out
}

/// Splits a byte stream (read in whatever pieces the socket gives) into messages.
#[derive(Default)]
pub struct Decoder {
    buf: Vec<u8>,
}

impl Decoder {
    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }
    /// The next whole message, `Err` when a length is over [`MAX_PAYLOAD`] (the stream is
    /// then unusable).
    pub fn next_message(&mut self) -> Result<Option<(u32, Vec<u8>)>, String> {
        if self.buf.len() < 8 {
            return Ok(None);
        }
        let kind = u32_at(&self.buf, 0);
        let len = u32_at(&self.buf, 4) as usize;
        if len > MAX_PAYLOAD {
            return Err(format!("message of {len} bytes"));
        }
        if self.buf.len() < 8 + len {
            return Ok(None);
        }
        let payload = self.buf[8..8 + len].to_vec();
        self.buf.drain(..8 + len);
        Ok(Some((kind, payload)))
    }
}

/// A named run of `f32`s in a vector.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub name: &'static str,
    pub len: usize,
}

const fn f(name: &'static str, len: usize) -> Field {
    Field { name, len }
}

/// The observation's state vector: what the policy sees (`game/rl/fair_play.rs` fills it).
/// Item fields hold the 26.3 item registry id + 1 (0 = empty); the registry's names come
/// with the welcome. Angles are degrees. Flags are 0 or 1.
pub const STATE: &[Field] = &[
    f("self_pos", 3),
    f("self_vel", 3),
    f("self_yaw", 1),
    f("self_pitch", 1),
    f("self_health", 1),
    f("self_absorption", 1),
    f("self_food", 1),
    f("self_armor", 1),
    f("self_attack_cooldown", 1),
    f("self_selected", 1),
    f("self_hotbar_item", 9),
    f("self_hotbar_count", 9),
    f("self_offhand_item", 1),
    f("self_offhand_count", 1),
    f("self_armor_item", 4),
    f("self_on_ground", 1),
    f("self_sprinting", 1),
    f("self_sneaking", 1),
    f("self_using_item", 1),
    f("self_hurt_time", 1),
    f("self_alive", 1),
    f("target_present", 1),
    f("target_rel_pos", 3),
    f("target_rel_vel", 3),
    f("target_distance", 1),
    f("target_yaw", 1),
    f("target_pitch", 1),
    // Where the target's eyes are from our view: yaw and pitch to turn by, in -180..180.
    f("target_bearing", 2),
    f("target_sneaking", 1),
    f("target_sprinting", 1),
    f("target_using_item", 1),
    f("target_swinging", 1),
    f("target_hurt_time", 1),
    f("target_on_fire", 1),
    f("target_main_hand", 1),
    f("target_off_hand", 1),
    f("target_armor_item", 4),
];

/// The info vector: signals for rewards and bookkeeping, never part of the observation.
/// Counts are for the step (the ticks since the last observation) unless marked cumulative.
pub const INFO: &[Field] = &[
    f("tick", 1),
    f("ticks_elapsed", 1),
    f("late_ticks", 1),
    f("tick_overrun_ms", 1),
    f("connected", 1),
    f("alive", 1),
    f("health", 1),
    f("damage_taken", 1),
    f("was_hit", 1),
    f("target_present", 1),
    // The target's synced health; NaN when the server never sent it.
    f("target_health", 1),
    f("target_damage", 1),
    f("target_hurt", 1),
    f("hit_landed", 1),
    f("attacks", 1),
    f("died", 1),
    f("respawned", 1),
    f("kills_total", 1),
    f("deaths_total", 1),
];

/// The number of `f32`s in `fields`.
pub const fn width(fields: &[Field]) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < fields.len() {
        n += fields[i].len;
        i += 1;
    }
    n
}

pub const STATE_LEN: usize = width(STATE);
pub const INFO_LEN: usize = width(INFO);

/// Where `name` starts in `fields`. Panics on an unknown name (a typo is a bug, not data).
pub fn offset(fields: &[Field], name: &str) -> usize {
    let mut at = 0;
    for field in fields {
        if field.name == name {
            return at;
        }
        at += field.len;
    }
    panic!("no field {name}")
}

/// Where each part of the shared-memory file is, in bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShmLayout {
    pub state_off: usize,
    pub info_off: usize,
    /// Images come with M3; 0 × 0 until then.
    pub frame_off: usize,
    pub frame_w: u32,
    pub frame_h: u32,
    pub size: usize,
}

impl ShmLayout {
    pub fn new(frame_w: u32, frame_h: u32) -> Self {
        let state_off = HEADER_LEN;
        let info_off = state_off + 4 * STATE_LEN;
        // Frames start on a 64-byte line.
        let frame_off = (info_off + 4 * INFO_LEN).div_ceil(64) * 64;
        let size = frame_off + (frame_w * frame_h) as usize;
        ShmLayout {
            state_off,
            info_off,
            frame_off,
            frame_w,
            frame_h,
            size,
        }
    }

    /// The header for an observation of step `seq` after tick `tick`.
    pub fn header(&self, seq: u64, tick: u64) -> [u8; HEADER_LEN] {
        let mut h = [0u8; HEADER_LEN];
        h[0..4].copy_from_slice(&SHM_MAGIC);
        h[4..8].copy_from_slice(&VERSION.to_le_bytes());
        h[8..16].copy_from_slice(&seq.to_le_bytes());
        h[16..24].copy_from_slice(&tick.to_le_bytes());
        for (i, v) in [
            self.state_off as u32,
            STATE_LEN as u32,
            self.info_off as u32,
            INFO_LEN as u32,
            self.frame_off as u32,
            self.frame_w,
            self.frame_h,
        ]
        .into_iter()
        .enumerate()
        {
            h[24 + 4 * i..28 + 4 * i].copy_from_slice(&v.to_le_bytes());
        }
        h
    }
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn fields_json(fields: &[Field]) -> String {
    let mut at = 0;
    let parts: Vec<String> = fields
        .iter()
        .map(|field| {
            let s = format!("[{},{},{}]", json_string(field.name), at, field.len);
            at += field.len;
            s
        })
        .collect();
    format!("[{}]", parts.join(","))
}

/// The `WELCOME` JSON: the version, the shm file and its layout, the state and info fields as
/// `[name, offset, len]` (in `f32`s), the item registry's names by id, and the player's name.
pub fn welcome(shm_path: &str, layout: &ShmLayout, items: &[&str], player: &str) -> String {
    let items: Vec<String> = items.iter().map(|n| json_string(n)).collect();
    format!(
        "{{\"version\":{VERSION},\"shm\":{},\"shm_size\":{},\"state_off\":{},\"info_off\":{},\"frame_off\":{},\"frame_w\":{},\"frame_h\":{},\"state\":{},\"info\":{},\"items\":[{}],\"player\":{}}}",
        json_string(shm_path),
        layout.size,
        layout.state_off,
        layout.info_off,
        layout.frame_off,
        layout.frame_w,
        layout.frame_h,
        fields_json(STATE),
        fields_json(INFO),
        items.join(","),
        json_string(player),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_bytes_are_little_endian_in_field_order() {
        // Hand-written: seq 0x0102, frame skip 3, forward + attack, slot 2, yaw 90, pitch -45.
        let mut bytes = vec![
            0x02, 0x01, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0x81, 0, 0, 0, 2, 0, 0, 0,
        ];
        bytes.extend_from_slice(&[0x00, 0x00, 0xb4, 0x42]); // 90.0f32
        bytes.extend_from_slice(&[0x00, 0x00, 0x34, 0xc2]); // -45.0f32
        let step = Step::decode(&bytes).unwrap();
        assert_eq!(
            step,
            Step {
                seq: 0x0102,
                frame_skip: 3,
                action: Action {
                    buttons: button::FORWARD | button::ATTACK,
                    hotbar: 2,
                    yaw: 90.0,
                    pitch: -45.0,
                },
            }
        );
        assert_eq!(step.encode(), bytes);
        assert!(Step::decode(&bytes[1..]).is_none());
    }

    #[test]
    fn decoder_reassembles_messages_split_anywhere() {
        let mut stream = frame(
            kind::HELLO,
            &Hello {
                version: 1,
                frame_skip: 2,
            }
            .encode(),
        );
        stream.extend(frame(kind::RESET, &[]));
        let mut d = Decoder::default();
        let mut got = Vec::new();
        for byte in &stream {
            d.push(std::slice::from_ref(byte));
            while let Some(m) = d.next_message().unwrap() {
                got.push(m);
            }
        }
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].0, kind::HELLO);
        assert_eq!(Hello::decode(&got[0].1).unwrap().frame_skip, 2);
        assert_eq!(got[1], (kind::RESET, vec![]));
    }

    #[test]
    fn oversized_lengths_are_refused() {
        let mut d = Decoder::default();
        d.push(&kind::STEP.to_le_bytes());
        d.push(&u32::MAX.to_le_bytes());
        assert!(d.next_message().is_err());
    }

    #[test]
    fn field_names_are_unique_and_offsets_contiguous() {
        for fields in [STATE, INFO] {
            let mut names: Vec<_> = fields.iter().map(|f| f.name).collect();
            names.sort();
            names.dedup();
            assert_eq!(names.len(), fields.len());
        }
        assert_eq!(offset(STATE, "self_pos"), 0);
        assert_eq!(offset(STATE, "self_vel"), 3);
        assert_eq!(
            offset(STATE, "target_armor_item") + 4,
            STATE_LEN,
            "the last field ends the vector"
        );
    }

    #[test]
    fn nothing_hidden_in_the_state_layout() {
        // The fair-play rule at the format level: the opponent's health, food, armor points and
        // effects have no place in the observation (they go to `info`, for rewards only).
        for field in STATE.iter().filter(|f| f.name.starts_with("target_")) {
            for hidden in [
                "health",
                "food",
                "absorption",
                "effect",
                "armor_value",
                "id",
            ] {
                assert!(
                    !field.name.contains(hidden),
                    "{} would leak {hidden}",
                    field.name
                );
            }
        }
    }

    #[test]
    fn shm_header_names_its_layout() {
        let l = ShmLayout::new(64, 48);
        let h = l.header(7, 99);
        assert_eq!(&h[0..4], b"HZRL");
        assert_eq!(u64_at(&h, 8), 7);
        assert_eq!(u64_at(&h, 16), 99);
        assert_eq!(u32_at(&h, 24) as usize, HEADER_LEN);
        assert_eq!(u32_at(&h, 28) as usize, STATE_LEN);
        assert_eq!(l.frame_off % 64, 0);
        assert!(l.frame_off >= l.info_off + 4 * INFO_LEN);
        assert_eq!(l.size, l.frame_off + 64 * 48);
    }

    #[test]
    fn welcome_is_json_with_escaped_names() {
        let w = welcome(
            "/dev/shm/a\"b",
            &ShmLayout::new(0, 0),
            &["minecraft:air"],
            "Bot_1",
        );
        assert!(w.contains("\"shm\":\"/dev/shm/a\\\"b\""));
        assert!(w.contains("[\"self_pos\",0,3]"));
        assert!(w.contains("\"items\":[\"minecraft:air\"]"));
    }
}
