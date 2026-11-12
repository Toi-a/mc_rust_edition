//! An RL client stays on its server (docs/rl/design.md#fault-isolation-and-reconnects): when the
//! connection is lost or a join fails, it joins again by itself after a backoff, instead of
//! waiting on the Disconnected screen for a click.
//!
//! Unlike the auto-reconnect extra (`reconnect.rs`, for people), every reason is retried, kicks
//! included: a training server kicks and restarts as part of its job, and no person is there to
//! click. The delays grow and stop growing at [`MAX_DELAY`]; a connection that lasted
//! [`STABLE`] starts them over.

use std::time::{Duration, Instant};

/// The first delay; each failed attempt doubles it up to [`MAX_DELAY`].
pub(super) const FIRST_DELAY: Duration = Duration::from_secs(1);
pub(super) const MAX_DELAY: Duration = Duration::from_secs(30);
/// Connected this long, the next loss counts as a first one.
pub(super) const STABLE: Duration = Duration::from_secs(30);

/// When to join again.
#[derive(Debug, Default)]
pub(super) struct Backoff {
    /// Attempts since the last stable connection.
    attempts: u32,
    /// The connection came up at this time and is still up.
    up_since: Option<Instant>,
    /// The next attempt, while the client is off its server.
    next: Option<Instant>,
}

impl Backoff {
    fn delay(attempts: u32) -> Duration {
        FIRST_DELAY
            .saturating_mul(1u32 << attempts.min(16))
            .min(MAX_DELAY)
    }

    /// The client is in a world on the server.
    pub(super) fn connected(&mut self, now: Instant) {
        self.next = None;
        let since = *self.up_since.get_or_insert(now);
        if now.saturating_duration_since(since) >= STABLE {
            self.attempts = 0;
        }
    }

    /// The client is off its server (the Disconnected or error screen). Returns true when it is
    /// time to join again; the attempt after it is planned at once, in case this one fails too.
    pub(super) fn off(&mut self, now: Instant) -> bool {
        self.up_since = None;
        let at = *self
            .next
            .get_or_insert_with(|| now + Self::delay(self.attempts));
        if now < at {
            return false;
        }
        self.attempts += 1;
        self.next = Some(now + Self::delay(self.attempts));
        true
    }

    /// Attempts since the last stable connection.
    pub(super) fn attempts(&self) -> u32 {
        self.attempts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secs(s: u64) -> Duration {
        Duration::from_secs(s)
    }

    /// The times at which a client that never gets back on joins again.
    fn attempts_over(seconds: u64) -> Vec<u64> {
        let start = Instant::now();
        let mut b = Backoff::default();
        (0..=seconds * 10)
            .map(|t| Duration::from_millis(t * 100))
            .filter(|&t| b.off(start + t))
            .map(|t| t.as_secs())
            .collect()
    }

    #[test]
    fn delays_double_and_stop_growing() {
        // 1, then +2, +4, +8, +16, then +30 each time.
        assert_eq!(attempts_over(130), vec![1, 3, 7, 15, 31, 61, 91, 121]);
    }

    #[test]
    fn a_stable_connection_starts_the_delays_over() {
        let t0 = Instant::now();
        let mut b = Backoff::default();
        assert!(!b.off(t0), "the first attempt waits FIRST_DELAY");
        assert!(b.off(t0 + secs(1)));
        assert!(b.off(t0 + secs(3)));
        assert_eq!(b.attempts(), 2);
        // Back on, but not for long: the next loss keeps counting.
        b.connected(t0 + secs(4));
        b.connected(t0 + secs(10));
        assert!(!b.off(t0 + secs(11)));
        assert!(b.off(t0 + secs(15)), "4 s after the loss (third attempt)");
        assert_eq!(b.attempts(), 3);
        // Back on for STABLE: a later loss waits the first delay only.
        b.connected(t0 + secs(20));
        b.connected(t0 + secs(20) + STABLE);
        assert_eq!(b.attempts(), 0);
        let lost = t0 + secs(100);
        assert!(!b.off(lost));
        assert!(b.off(lost + FIRST_DELAY));
    }
}
