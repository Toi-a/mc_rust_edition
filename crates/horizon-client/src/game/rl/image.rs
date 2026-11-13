//! Image observations (`--rl-image WxH`, docs/rl/design.md "Images"): the player's normal view,
//! drawn offscreen at W x H, turned into one luma byte per pixel on the GPU
//! (`gfx::luma::LumaCapture`) and copied into the shared memory's frame slot.
//!
//! - A step's observation is written after its last tick, so its frame is the one drawn right
//!   after that tick (`Game::rl_render_frame`): the frame and the state show the same tick.
//!   Frames are drawn only for observations: with frame skip k, one in k ticks; between them
//!   the headless game draws nothing.
//! - The HUD is hidden unless `--rl-hud` (vanilla's F1, which also hides the hand and every
//!   name tag); nothing extra is ever drawn.
//! - A frame that couldn't be drawn (no renderer, an error) leaves the last one in place, and
//!   the info's `frame_fresh` is 0.

use super::super::super::{Game, Result};
use super::Session;
use horizon_client::gfx::luma::LumaCapture;
use std::{os::unix::fs::FileExt, time::Instant};

/// The capture's state, with `--rl-image`.
pub(super) struct Frames {
    capture: Option<LumaCapture>,
    bytes: Vec<u8>,
    /// An observation waits for this frame: the seq of its step.
    pub(super) due: Option<u64>,
    /// The frame of the last observation was drawn for it.
    pub(super) fresh: bool,
    /// Drawing and reading it back took this long (ms).
    pub(super) ms: f32,
}

impl Frames {
    pub(super) fn new([w, h]: [u32; 2]) -> Self {
        Frames {
            capture: None,
            bytes: vec![0; (w * h) as usize],
            due: None,
            fresh: false,
            ms: 0.0,
        }
    }
}

/// `--rl-image WxH`: the width a multiple of 4 (four pixels per word in the GPU pass), both
/// 8..=512.
pub(in super::super::super) fn parse_image_size(
    text: &str,
) -> std::result::Result<[u32; 2], String> {
    let (w, h) = text
        .split_once(['x', 'X'])
        .ok_or_else(|| format!("--rl-image {text}: expected WxH, e.g. 64x64"))?;
    let side = |s: &str| {
        s.trim()
            .parse::<u32>()
            .ok()
            .filter(|v| (8..=512).contains(v))
            .ok_or_else(|| format!("--rl-image {text}: sides go from 8 to 512"))
    };
    let (w, h) = (side(w)?, side(h)?);
    if !w.is_multiple_of(4) {
        return Err(format!(
            "--rl-image {text}: the width must be a multiple of 4"
        ));
    }
    Ok([w, h])
}

impl Game<'_> {
    /// The frame (`Game::render`'s, apart from screenshot sequences). With image observations,
    /// a frame is drawn only when an observation waits for it, and then read back into the
    /// shared memory before the observation goes out.
    pub(in super::super::super) fn rl_render_frame(&mut self) -> Result<bool> {
        let due = match self.agent.rl.as_deref() {
            Some(Session {
                frames: Some(f), ..
            }) => Some(f.due),
            _ => None,
        };
        match due {
            None => self.render_frame(),
            Some(None) if self.options.headless => Ok(false),
            Some(None) => self.render_frame(),
            Some(Some(seq)) => {
                let started = Instant::now();
                let drawn = self.render_frame();
                let mut s = self.agent.rl.take().expect("checked above");
                let fresh = matches!(drawn, Ok(true)) && self.rl_capture(&mut s);
                if let Some(f) = s.frames.as_mut() {
                    f.fresh = fresh;
                    f.ms = started.elapsed().as_secs_f32() * 1000.0;
                    f.due = None;
                }
                self.rl_write(&mut s, seq);
                self.agent.rl = Some(s);
                drawn
            }
        }
    }

    /// Reads the frame just drawn into the shared memory's frame slot.
    fn rl_capture(&mut self, s: &mut Session) -> bool {
        let (Some(renderer), Some(f)) = (self.renderer.as_mut(), s.frames.as_mut()) else {
            return false;
        };
        if renderer.offscreen_size() != Some([s.layout.frame_w, s.layout.frame_h]) {
            // The game resized the picture (it shouldn't, headless): draw at the asked size.
            renderer.resize(s.layout.frame_w, s.layout.frame_h);
            return false;
        }
        if let Err(e) = renderer.capture_luma(&mut f.capture, &mut f.bytes) {
            log::warn!("RL frame: {e}");
            return false;
        }
        if let Err(e) = s.shm.write_all_at(&f.bytes, s.layout.frame_off as u64) {
            log::error!("RL shm: {e}");
            return false;
        }
        true
    }

    /// Builds the headless game's renderer (`--headless --render`): no window, an offscreen
    /// picture of `--width` x `--height`.
    pub(in super::super::super) fn headless_renderer(&mut self) {
        if !(self.options.headless && self.options.render) || self.renderer.is_some() {
            return;
        }
        self.gui.no_title_panorama();
        match self.renderer_for(
            None::<std::sync::Arc<winit::window::Window>>,
            &self.pack,
            &self.gui,
        ) {
            Ok(mut renderer) => {
                let [w, h] = self.options.window_size;
                renderer.resize(w, h);
                self.renderer = Some(renderer);
                self.world_renderer_ready = true;
                self.apply_settings();
                log::info!("Headless renderer: {w}x{h} offscreen");
            }
            Err(e) => log::error!("Headless renderer: {e}; no frames will be drawn"),
        }
    }
}

impl super::super::super::UiResources {
    /// A headless game opens no title screen: its panorama becomes one texel, instead of
    /// six decoded faces (36 MB) waiting for the first title screen to take them.
    pub(in super::super::super) fn no_title_panorama(&mut self) {
        self.panorama = std::sync::Arc::new(horizon_client::gfx::panorama::Faces::colour([0; 4]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_sizes_are_wxh_with_a_width_in_fours() {
        assert_eq!(parse_image_size("64x64"), Ok([64, 64]));
        assert_eq!(parse_image_size("128X96"), Ok([128, 96]));
        assert!(
            parse_image_size("66x64").is_err(),
            "width not a multiple of 4"
        );
        assert!(parse_image_size("64").is_err());
        assert!(parse_image_size("4x4").is_err(), "too small");
        assert!(parse_image_size("1024x64").is_err(), "too large");
    }
}
