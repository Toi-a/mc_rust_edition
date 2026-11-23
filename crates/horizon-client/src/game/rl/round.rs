//! Rounds of the 1v1, run by the training server's plugin
//! (docs/rl/design.md#rounds-and-the-opponent-pool-m6-planned).
//!
//! The plugin tells each agent over the plugin channel [`CHANNEL`] when its pair's round starts
//! and how it ended for this agent, one UTF-8 JSON object per message:
//!
//! - `{"v":1,"type":"start","round":12,"opponent":"Agent_2","max_ticks":2400}`
//! - `{"v":1,"type":"end","round":12,"result":"win|loss|draw|timeout|abort","reason":"...","ticks":345}`
//!
//! Bukkit sends a plugin message only to clients that announced its channel, so the session
//! announces it (`minecraft:register`) at each login. The messages are read in packet order,
//! among the others (the reset's teleport and health come before the start), and become the
//! round signals in `info` and the time left in the state (the cap is a rule of the server, so
//! its players know it).

use super::super::super::Game;
use super::Signals;
use horizon_protocol::packets::{
    configuration::CustomPayload,
    play::{Clientbound as C, Serverbound as S},
};

/// The plugin channel of the round messages.
pub(super) const CHANNEL: &str = "horizon_rl:round";
/// The version of the messages this client reads.
const VERSION: u64 = 1;

/// How a round ended for this agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    Win,
    Loss,
    /// Both agents died on the same tick.
    Draw,
    /// The time cap ran out with no death.
    Timeout,
    /// The opponent left during the fight, or the server stopped the round.
    Abort,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Message {
    Start {
        round: u32,
        opponent: String,
        /// The time cap in server ticks; 0 = none.
        max_ticks: u32,
    },
    End {
        round: u32,
        outcome: Outcome,
        /// Server ticks from the round's start to its end.
        ticks: u32,
    },
}

/// One message's payload.
pub(super) fn parse(data: &[u8]) -> Result<Message, String> {
    let v: serde_json::Value = serde_json::from_slice(data).map_err(|e| e.to_string())?;
    let version = v.get("v").and_then(|v| v.as_u64());
    if version != Some(VERSION) {
        return Err(format!("version {version:?}, this client reads {VERSION}"));
    }
    let number = |key: &str| {
        v.get(key)
            .and_then(|n| n.as_u64())
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| format!("no {key}"))
    };
    let text = |key: &str| {
        v.get(key)
            .and_then(|s| s.as_str())
            .ok_or_else(|| format!("no {key}"))
    };
    match text("type")? {
        "start" => Ok(Message::Start {
            round: number("round")?,
            opponent: text("opponent")?.to_owned(),
            max_ticks: number("max_ticks")?,
        }),
        "end" => Ok(Message::End {
            round: number("round")?,
            outcome: match text("result")? {
                "win" => Outcome::Win,
                "loss" => Outcome::Loss,
                "draw" => Outcome::Draw,
                "timeout" => Outcome::Timeout,
                "abort" => Outcome::Abort,
                other => return Err(format!("result {other:?}")),
            },
            ticks: number("ticks")?,
        }),
        other => Err(format!("type {other:?}")),
    }
}

/// The round this client is in, kept across steps and resets.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Round {
    /// The last round started or ended (0: none yet).
    pub id: u32,
    /// The session tick the running round started at; `None` between rounds.
    pub started: Option<u64>,
    pub max_ticks: u32,
    /// A start named an opponent other than `--rl-target` (warned once).
    pub warned_opponent: bool,
}

impl Round {
    /// The share of the time cap left, 1 at the start down to 0; 1 throughout a round without a
    /// cap, 0 between rounds.
    pub(super) fn time_left(&self, now: u64) -> f32 {
        let Some(started) = self.started else {
            return 0.0;
        };
        if self.max_ticks == 0 {
            return 1.0;
        }
        let used = now.saturating_sub(started) as f32 / self.max_ticks as f32;
        (1.0 - used).clamp(0.0, 1.0)
    }
}

/// A round message from the server at session tick `now`, into the round and the step's
/// signals. `target` is `--rl-target`.
fn apply(round: &mut Round, s: &mut Signals, target: Option<&str>, now: u64, message: Message) {
    match message {
        Message::Start {
            round: id,
            opponent,
            max_ticks,
        } => {
            if let Some(target) = target.filter(|t| *t != opponent)
                && !round.warned_opponent
            {
                log::warn!(
                    "RL: the server's round names {opponent} as the opponent, --rl-target is {target}"
                );
                round.warned_opponent = true;
            }
            round.id = id;
            round.started = Some(now);
            round.max_ticks = max_ticks;
            s.round_start += 1;
        }
        Message::End {
            round: id,
            outcome,
            ticks,
        } => {
            round.id = id;
            round.started = None;
            s.round_end += 1;
            s.round_ticks = ticks as f32;
            *match outcome {
                Outcome::Win => &mut s.round_win,
                Outcome::Loss => &mut s.round_loss,
                Outcome::Draw => &mut s.round_draw,
                Outcome::Timeout => &mut s.round_timeout,
                Outcome::Abort => &mut s.round_abort,
            } += 1;
        }
    }
}

impl Game<'_> {
    /// Every play packet from the server, before the game handles it (`handle_packet`): the
    /// native mods' hook, then the RL session's rounds.
    pub(in super::super::super) fn client_packet_hooks(&mut self, packet: &C) {
        self.native_client_packet(packet);
        if self.agent.rl.is_some() {
            self.rl_round_packet(packet);
        }
    }

    fn rl_round_packet(&mut self, packet: &C) {
        match packet {
            // Announce the channel at each login: Bukkit sends nothing on one a client didn't.
            C::Login(_) => self.send(S::CustomPayload(CustomPayload {
                channel: "minecraft:register".into(),
                data: CHANNEL.as_bytes().to_vec(),
            })),
            C::CustomPayload(p) if p.channel == CHANNEL => match parse(&p.data) {
                Ok(message) => {
                    log::debug!("RL round: {message:?}");
                    if let Some(s) = self.agent.rl.as_mut() {
                        let target = s.options.target.as_deref();
                        apply(&mut s.round, &mut s.signals, target, s.ticks, message);
                    }
                }
                Err(e) => log::warn!(
                    "RL: a round message this client can't read ({e}): {}",
                    String::from_utf8_lossy(&p.data)
                ),
            },
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_plugins_messages() {
        let start = br#"{"v":1,"type":"start","round":12,"opponent":"Agent_2","max_ticks":2400}"#;
        assert_eq!(
            parse(start),
            Ok(Message::Start {
                round: 12,
                opponent: "Agent_2".into(),
                max_ticks: 2400
            })
        );
        for (result, outcome) in [
            ("win", Outcome::Win),
            ("loss", Outcome::Loss),
            ("draw", Outcome::Draw),
            ("timeout", Outcome::Timeout),
            ("abort", Outcome::Abort),
        ] {
            let end = format!(
                r#"{{"v":1,"type":"end","round":12,"result":"{result}","reason":"kill","ticks":345}}"#
            );
            assert_eq!(
                parse(end.as_bytes()),
                Ok(Message::End {
                    round: 12,
                    outcome,
                    ticks: 345
                })
            );
        }
    }

    #[test]
    fn refuses_what_it_cant_read() {
        for bad in [
            &br#"{"v":2,"type":"start","round":1,"opponent":"A","max_ticks":0}"#[..],
            br#"{"type":"start","round":1,"opponent":"A","max_ticks":0}"#,
            br#"{"v":1,"type":"pause","round":1}"#,
            br#"{"v":1,"type":"end","round":1,"result":"won","ticks":3}"#,
            br#"{"v":1,"type":"end","round":-1,"result":"win","ticks":3}"#,
            br#"{"v":1,"type":"start","round":1,"max_ticks":0}"#,
            b"not json",
        ] {
            assert!(parse(bad).is_err(), "{}", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn a_round_starts_then_ends_in_the_signals() {
        let (mut round, mut s) = (Round::default(), Signals::default());
        let start = Message::Start {
            round: 3,
            opponent: "Agent_2".into(),
            max_ticks: 2400,
        };
        apply(&mut round, &mut s, Some("Agent_2"), 40, start);
        assert_eq!(
            (round.id, round.started, round.max_ticks),
            (3, Some(40), 2400)
        );
        assert_eq!((s.round_start, s.round_end), (1, 0));
        assert!(!round.warned_opponent);
        let end = Message::End {
            round: 3,
            outcome: Outcome::Timeout,
            ticks: 2400,
        };
        apply(&mut round, &mut s, Some("Agent_2"), 2440, end);
        assert_eq!((round.id, round.started), (3, None));
        assert_eq!(
            (s.round_end, s.round_timeout, s.round_ticks),
            (1, 1, 2400.0)
        );
        assert_eq!(s.round_win + s.round_loss + s.round_draw + s.round_abort, 0);
    }

    #[test]
    fn time_left_runs_down_over_the_cap() {
        let mut r = Round::default();
        assert_eq!(r.time_left(5), 0.0, "between rounds");
        r.started = Some(100);
        r.max_ticks = 2400;
        assert_eq!(r.time_left(100), 1.0);
        assert_eq!(r.time_left(700), 0.75);
        assert_eq!(r.time_left(2500), 0.0);
        assert_eq!(r.time_left(9000), 0.0, "never below 0");
        r.max_ticks = 0;
        assert_eq!(r.time_left(9000), 1.0, "no cap");
    }
}
