//! Read-only textures shared by the renderers of one process (RL process model B,
//! docs/rl/design.md "Model B measurements"): K headless clients on one device would otherwise each
//! upload the same pack, ~126 of a 64x64 image client's ~170 MB of GPU memory. A sibling
//! renderer, made with `previous` set while [`Siblings`] is held, takes the previous renderer's
//! shared atlas (entities, items, GUI: never written after it is made) and block sprites. The
//! block sprites' animated layers are then written by every sibling with its own tick: the
//! frames of water or fire may come from another client's tick, which a policy can't tell.
//! A pack reload (`previous` without the guard) still builds its own.

use super::Renderer;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};

static SIBLINGS: AtomicBool = AtomicBool::new(false);

/// While it lives, a renderer made with `previous` shares that renderer's read-only textures:
/// only for renderers built from the same packs (the clients of one RL process).
pub struct Siblings(());

impl Siblings {
    pub fn hold() -> Self {
        SIBLINGS.store(true, Ordering::SeqCst);
        Self(())
    }
}

impl Drop for Siblings {
    fn drop(&mut self) {
        SIBLINGS.store(false, Ordering::SeqCst);
    }
}

pub(super) fn sibling(previous: Option<&Renderer>) -> Option<&Renderer> {
    previous.filter(|_| SIBLINGS.load(Ordering::SeqCst))
}

/// The sibling's block sprite arrays, and their mip level count in `mip_levels`.
pub(super) fn sprites(
    sibling: Option<&Renderer>,
    mip_levels: &mut u32,
) -> Option<Vec<wgpu::Texture>> {
    let textures = sibling?.sprite_textures.clone();
    *mip_levels = textures.first()?.mip_level_count();
    Some(textures)
}

/// The sibling's shared atlas.
pub(super) fn atlas(sibling: Option<&Renderer>) -> Option<wgpu::Texture> {
    sibling?.hud.atlas_texture.clone()
}

impl Renderer {
    /// Device memory the backend's allocator holds, by the label of what holds
    /// it, largest first, with its allocated and reserved totals (Vulkan and
    /// DirectX 12; none on Metal and OpenGL) (`horizon-ctl memory`). Siblings share
    /// one device, so each reports what all of them hold.
    pub fn gpu_memory_by_label(&self) -> Option<(u64, u64, Vec<(String, u64)>)> {
        let report = self.device.generate_allocator_report()?;
        let mut by_label: HashMap<String, u64> = HashMap::new();
        for a in report.allocations {
            *by_label.entry(a.name).or_default() += a.size;
        }
        let mut labels: Vec<_> = by_label.into_iter().collect();
        labels.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Some((
            report.total_allocated_bytes,
            report.total_reserved_bytes,
            labels,
        ))
    }
}
