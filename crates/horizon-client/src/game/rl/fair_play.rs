//! What the policy may see: the only code that reads `Game` for an RL observation
//! (docs/rl/design.md#what-the-agent-observes-one-place-gamerlfair_playrs).
//!
//! The rule: the policy may know what a person playing on the same server knows. The base is a
//! vanilla client; what the training server shows its players on top of that is added per run,
//! by a switch named here, never by default.
//!
//! The setting is a 1v1 duel on flat ground, so there is no visibility test: the agent's one
//! opponent (its *target*) is observed whenever this client tracks it. The rules:
//!
//! - **Self**: what the HUD and F3 show about yourself (position, velocity, view, health,
//!   absorption, food, armor bar, attack cooldown, hotbar, off hand, worn armor, movement flags,
//!   using an item, the hurt flash). Saturation is out: vanilla doesn't show it.
//! - **Target**: what you see of another player: where it is and how it moves, where it looks,
//!   its pose (sneaking, sprinting, using an item, swinging), the hurt flash, fire, and what it
//!   holds and wears.
//! - **Only where the server shows it** (`--rl-target-health`): the target's health and
//!   absorption. PvP servers draw every player's health above the name, so there a person knows
//!   it. Without the switch [`TargetView::shown_health`] is `None` and both fields stay zero; the
//!   target's health still goes to `info` (rewards only) by a separate path in `mod.rs`.
//! - **Never**: the target's food, armor points or effects, or any network entity id.
//!   [`TargetView`] has no field for them, so [`encode`] can't write them.
//! - No other entity: one opponent, no block grid.

use horizon_client::mobs::Mob;
use horizon_client::pose::Pose;
use horizon_rl_proto::{STATE, STATE_LEN, offset};

/// The agent itself, as its HUD and F3 screen show it.
#[derive(Clone, Debug, Default)]
pub(super) struct SelfView {
    pub alive: bool,
    pub pos: [f64; 3],
    pub vel: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub health: f32,
    pub absorption: f32,
    pub food: i32,
    pub armor: i32,
    pub attack_cooldown: f32,
    pub selected: usize,
    /// Item id + 1 (0 = empty) and count, slots 1..9.
    pub hotbar: [(u32, u32); 9],
    pub offhand: (u32, u32),
    /// Head, chest, legs, feet.
    pub armor_items: [u32; 4],
    pub on_ground: bool,
    pub sprinting: bool,
    pub sneaking: bool,
    pub using_item: bool,
    pub hurt_time: i32,
}

/// What the agent sees of its target. By construction it carries nothing a player can't see.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct TargetView {
    pub pos: [f64; 3],
    /// Blocks per tick, from this client's own sightings (position deltas).
    pub vel: [f64; 3],
    /// Where its head points.
    pub yaw: f32,
    pub pitch: f32,
    pub eye_height: f64,
    pub sneaking: bool,
    pub sprinting: bool,
    pub using_item: bool,
    pub swinging: bool,
    pub hurt_time: i32,
    pub on_fire: bool,
    pub main_hand: u32,
    pub off_hand: u32,
    pub armor_items: [u32; 4],
    /// Health and absorption, only on a server that shows them to its players
    /// (`--rl-target-health`).
    pub shown_health: Option<(f32, f32)>,
}

impl TargetView {
    /// What is visible of `m`. `item` turns an item name into its id + 1 (0 = none);
    /// `previous` is where this client saw it last tick, for its velocity; `health_shown` says
    /// the server shows its players each other's health.
    pub(super) fn of(
        m: &Mob,
        previous: Option<[f64; 3]>,
        item: impl Fn(Option<&str>) -> u32,
        health_shown: bool,
    ) -> Self {
        let pos = [m.position.x, m.position.y, m.position.z];
        let vel = seen_velocity(pos, previous);
        let e = &m.extra;
        TargetView {
            pos,
            vel,
            yaw: m.head_yaw,
            pitch: m.head_pitch,
            eye_height: m.kind.eye_height,
            sneaking: e.avatar_pose == Pose::Crouching,
            sprinting: e.sprinting,
            using_item: e.using_item != 0,
            swinging: m.ai.swinging,
            hurt_time: m.hurt_time,
            on_fire: m.fire_ticks > 0,
            main_hand: item(e.main_hand.as_deref()),
            off_hand: item(e.off_hand.as_deref()),
            armor_items: [
                item(e.head.as_deref()),
                item(e.chest.as_deref()),
                item(e.legs.as_deref()),
                item(e.feet.as_deref()),
            ],
            shown_health: health_shown.then_some((m.health, m.absorption)),
        }
    }
}

/// Blocks per tick between two sightings; zero on the first.
pub(super) fn seen_velocity(pos: [f64; 3], previous: Option<[f64; 3]>) -> [f64; 3] {
    previous.map_or([0.0; 3], |p| [pos[0] - p[0], pos[1] - p[1], pos[2] - p[2]])
}

fn flag(b: bool) -> f32 {
    if b { 1.0 } else { 0.0 }
}

/// An angle folded into -180..180.
pub(super) fn wrap_degrees(a: f64) -> f64 {
    let a = a.rem_euclid(360.0);
    if a >= 180.0 { a - 360.0 } else { a }
}

/// The eye height of a standing player (`Player.STANDING_DIMENSIONS`' eye height).
const PLAYER_EYE: f64 = 1.62;
const SNEAK_EYE: f64 = 1.27;

/// Writes the observation's state vector. Fields left out (no target) are zero.
pub(super) fn encode(me: &SelfView, target: Option<&TargetView>, out: &mut [f32; STATE_LEN]) {
    out.fill(0.0);
    let mut put = |name: &str, values: &[f32]| {
        let at = offset(STATE, name);
        out[at..at + values.len()].copy_from_slice(values);
    };
    let f3 = |v: [f64; 3]| [v[0] as f32, v[1] as f32, v[2] as f32];
    put("self_pos", &f3(me.pos));
    put("self_vel", &f3(me.vel));
    put("self_yaw", &[me.yaw]);
    put("self_pitch", &[me.pitch]);
    put("self_health", &[me.health]);
    put("self_absorption", &[me.absorption]);
    put("self_food", &[me.food as f32]);
    put("self_armor", &[me.armor as f32]);
    put("self_attack_cooldown", &[me.attack_cooldown]);
    put("self_selected", &[me.selected as f32]);
    put("self_hotbar_item", &me.hotbar.map(|(item, _)| item as f32));
    put(
        "self_hotbar_count",
        &me.hotbar.map(|(_, count)| count as f32),
    );
    put("self_offhand_item", &[me.offhand.0 as f32]);
    put("self_offhand_count", &[me.offhand.1 as f32]);
    put("self_armor_item", &me.armor_items.map(|i| i as f32));
    put("self_on_ground", &[flag(me.on_ground)]);
    put("self_sprinting", &[flag(me.sprinting)]);
    put("self_sneaking", &[flag(me.sneaking)]);
    put("self_using_item", &[flag(me.using_item)]);
    put("self_hurt_time", &[me.hurt_time as f32]);
    put("self_alive", &[flag(me.alive)]);
    let Some(t) = target else {
        return;
    };
    let rel = [
        t.pos[0] - me.pos[0],
        t.pos[1] - me.pos[1],
        t.pos[2] - me.pos[2],
    ];
    let rel_vel = [
        t.vel[0] - me.vel[0],
        t.vel[1] - me.vel[1],
        t.vel[2] - me.vel[2],
    ];
    put("target_present", &[1.0]);
    put("target_rel_pos", &f3(rel));
    put("target_rel_vel", &f3(rel_vel));
    put(
        "target_distance",
        &[(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]).sqrt() as f32],
    );
    put("target_yaw", &[t.yaw]);
    put("target_pitch", &[t.pitch]);
    // From our eyes to its eyes, as Minecraft's yaw (0 = +z, 90 = -x) and pitch (down +).
    let my_eye = me.pos[1] + if me.sneaking { SNEAK_EYE } else { PLAYER_EYE };
    let dx = t.pos[0] - me.pos[0];
    let dy = t.pos[1] + t.eye_height - my_eye;
    let dz = t.pos[2] - me.pos[2];
    let yaw_to = (-dx).atan2(dz).to_degrees();
    let pitch_to = (-dy).atan2(dx.hypot(dz)).to_degrees();
    put(
        "target_bearing",
        &[
            wrap_degrees(yaw_to - f64::from(me.yaw)) as f32,
            (pitch_to - f64::from(me.pitch)) as f32,
        ],
    );
    put("target_sneaking", &[flag(t.sneaking)]);
    put("target_sprinting", &[flag(t.sprinting)]);
    put("target_using_item", &[flag(t.using_item)]);
    put("target_swinging", &[flag(t.swinging)]);
    put("target_hurt_time", &[t.hurt_time as f32]);
    put("target_on_fire", &[flag(t.on_fire)]);
    put("target_main_hand", &[t.main_hand as f32]);
    put("target_off_hand", &[t.off_hand as f32]);
    put("target_armor_item", &t.armor_items.map(|i| i as f32));
    if let Some((health, absorption)) = t.shown_health {
        put("target_health", &[health]);
        put("target_absorption", &[absorption]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn me() -> SelfView {
        SelfView {
            alive: true,
            pos: [0.0, 64.0, 0.0],
            health: 20.0,
            ..Default::default()
        }
    }

    fn opponent_at(pos: [f64; 3]) -> TargetView {
        TargetView {
            pos,
            eye_height: PLAYER_EYE,
            main_hand: 1001,
            ..Default::default()
        }
    }

    fn observe(t: &TargetView) -> [f32; STATE_LEN] {
        let mut out = [0.0; STATE_LEN];
        encode(&me(), Some(t), &mut out);
        out
    }

    #[test]
    fn the_opponent_is_seen_where_it_stands_and_with_what_it_holds() {
        let out = observe(&opponent_at([3.0, 64.0, 4.0]));
        let at = |name| out[offset(STATE, name)];
        assert_eq!(at("target_present"), 1.0);
        let rel = offset(STATE, "target_rel_pos");
        assert_eq!(&out[rel..rel + 3], &[3.0, 0.0, 4.0]);
        assert_eq!(at("target_distance"), 5.0);
        assert_eq!(at("target_main_hand"), 1001.0);
    }

    #[test]
    fn the_opponents_health_is_seen_only_where_the_server_shows_it() {
        let hidden = observe(&opponent_at([3.0, 64.0, 4.0]));
        let at = |out: &[f32; STATE_LEN], name| out[offset(STATE, name)];
        assert_eq!(at(&hidden, "target_health"), 0.0);
        assert_eq!(at(&hidden, "target_absorption"), 0.0);
        let shown = observe(&TargetView {
            shown_health: Some((13.5, 4.0)),
            ..opponent_at([3.0, 64.0, 4.0])
        });
        assert_eq!(at(&shown, "target_health"), 13.5);
        assert_eq!(at(&shown, "target_absorption"), 4.0);
    }

    #[test]
    fn no_target_leaves_every_target_field_zero() {
        let mut out = [1.0; STATE_LEN];
        encode(&me(), None, &mut out);
        for field in STATE.iter().filter(|f| f.name.starts_with("target_")) {
            let at = offset(STATE, field.name);
            assert!(
                out[at..at + field.len].iter().all(|v| *v == 0.0),
                "{}",
                field.name
            );
        }
        assert_eq!(out[offset(STATE, "self_health")], 20.0);
    }

    #[test]
    fn bearing_says_how_far_to_turn_to_face_the_opponent() {
        // At the origin facing south (+z, yaw 0); the opponent stands due west (-x), which
        // Minecraft calls yaw 90, at the same eye height.
        let out = observe(&opponent_at([-5.0, 64.0, 0.0]));
        let b = offset(STATE, "target_bearing");
        assert!((out[b] - 90.0).abs() < 1e-4, "yaw {}", out[b]);
        assert!(out[b + 1].abs() < 1e-4, "pitch {}", out[b + 1]);
        // Behind us (north): half a turn either way.
        let out = observe(&opponent_at([0.0, 64.0, -5.0]));
        assert!((out[b].abs() - 180.0).abs() < 1e-4, "yaw {}", out[b]);
    }

    #[test]
    fn velocity_comes_from_what_this_client_saw_move() {
        assert_eq!(
            seen_velocity([3.0, 64.0, 4.0], Some([2.5, 64.0, 4.0])),
            [0.5, 0.0, 0.0]
        );
        assert_eq!(
            seen_velocity([3.0, 64.0, 4.0], None),
            [0.0; 3],
            "a first sighting"
        );
    }

    #[test]
    fn angles_wrap_into_half_turns() {
        assert_eq!(wrap_degrees(190.0), -170.0);
        assert_eq!(wrap_degrees(-190.0), 170.0);
        assert_eq!(wrap_degrees(720.0), 0.0);
    }
}
