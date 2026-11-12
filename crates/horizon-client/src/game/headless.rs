//! `--headless` (#4951): a client with no display, no window and no renderer.
//!
//! winit's event loop needs a display server even when no window is shown, so a headless game
//! runs this loop instead. Each step is what winit's `about_to_wait` does (`Game::idle`: stop
//! requests, the closing save, world entry, the control socket), and `idle` runs the frame
//! (the client tick, nothing drawn) through the same `RedrawRequested` path a window's game
//! takes. Input comes only from the control socket (`horizon-ctl`), as synthetic window events.

use super::Game;
use super::event_loop::EventLoopCtl;
use std::{
    cell::Cell,
    time::{Duration, Instant},
};
use winit::event_loop::ControlFlow;

/// Time between headless frames. The client ticks at 20 Hz whatever the frame rate (a frame
/// runs the ticks its elapsed time owes), so one frame per tick does all the work there is
/// without spending a core on frames nobody sees.
const FRAME: Duration = Duration::from_millis(50);
/// In a world, a frame comes this long after its tick is due, so it always finds the tick owed.
const PAST_DUE: Duration = Duration::from_micros(500);

/// The loop's side of `EventLoopCtl`: what the game asked for during a step.
struct HeadlessLoop {
    exit: Cell<bool>,
    flow: Cell<ControlFlow>,
}

impl EventLoopCtl for HeadlessLoop {
    fn exit(&self) {
        self.exit.set(true);
    }
    fn set_control_flow(&self, flow: ControlFlow) {
        self.flow.set(flow);
    }
}

impl HeadlessLoop {
    /// When the step that started at `started` wants the next one: at the time the game asked
    /// for (`WaitUntil`, as the closing save's polls do), else when the next tick is due
    /// (`tick_due`, in a world), else one frame later.
    fn next_step(&self, started: Instant, tick_due: Option<Instant>) -> Instant {
        match self.flow.get() {
            ControlFlow::WaitUntil(at) => at,
            ControlFlow::Poll | ControlFlow::Wait => tick_due.unwrap_or(started + FRAME),
        }
    }
}

impl Game<'_> {
    /// Runs the game until it asks to stop (`horizon-ctl quit`, a signal, an error), then
    /// saves as a window's game does when its loop ends.
    pub(super) fn run_headless(&mut self) {
        let ctl = HeadlessLoop {
            exit: Cell::new(false),
            flow: Cell::new(ControlFlow::Poll),
        };
        // `--render`: a renderer without a window (RL image observations).
        self.headless_renderer();
        self.previous = Instant::now();
        while !ctl.exit.get() {
            // An RL client lines its ticks up after its steps (game/rl): the wait moves the
            // frame and the tick clock together.
            let waited = self.rl_await_step();
            self.previous += waited;
            let started = Instant::now();
            ctl.flow.set(ControlFlow::Poll);
            self.idle(&ctl);
            if let Some(wait) = ctl
                .next_step(started, self.tick_due(Instant::now()))
                .checked_duration_since(Instant::now())
            {
                std::thread::sleep(wait);
            }
        }
        self.shut_down();
    }

    /// When the tick clock's next tick is due, in a world that ticks: from the residual the last
    /// frame left. Waking then gives every frame one tick, where a fixed 50 ms cadence gives a
    /// frame with none and one with two whenever the residual sits on a tick's edge (and the
    /// second of two ticks can't get a fresh RL step).
    fn tick_due(&self, now: Instant) -> Option<Instant> {
        if self.title.is_some() || self.pause || !self.clock.runs_normally() {
            return None;
        }
        let left = self
            .clock
            .tick_duration()
            .mul_f64((1.0 - self.clock.fraction()).clamp(0.0, 1.0));
        Some(now + left.min(FRAME) + PAST_DUE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctl(flow: ControlFlow) -> HeadlessLoop {
        HeadlessLoop {
            exit: Cell::new(false),
            flow: Cell::new(flow),
        }
    }

    #[test]
    fn steps_come_a_frame_apart_unless_the_game_asks_for_sooner() {
        let started = Instant::now();
        assert_eq!(
            ctl(ControlFlow::Poll).next_step(started, None),
            started + FRAME
        );
        let soon = started + Duration::from_millis(4);
        assert_eq!(
            ctl(ControlFlow::WaitUntil(soon)).next_step(started, None),
            soon
        );
        let due = started + Duration::from_millis(30);
        assert_eq!(ctl(ControlFlow::Poll).next_step(started, Some(due)), due);
        assert_eq!(
            ctl(ControlFlow::WaitUntil(soon)).next_step(started, Some(due)),
            soon
        );
    }

    #[test]
    fn exit_is_remembered_for_the_loop() {
        let c = ctl(ControlFlow::Poll);
        assert!(!c.exit.get());
        EventLoopCtl::exit(&c);
        assert!(c.exit.get());
    }
}
