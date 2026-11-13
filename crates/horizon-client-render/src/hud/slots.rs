//! The atlas's slots for images made at run time, and the lean atlas of a game that draws the
//! world but never its GUI (an RL image client without `--rl-hud`, docs/rl/design.md "M5
//! measurements"): `HORIZON_LEAN_ATLAS=1` leaves the GUI's textures and the menus' slots out of
//! the pixels. The default resources' atlas is then 2048x3968 (31 MiB) instead of 4096x6144
//! (96 MiB).
//!
//! A lean atlas still names every sprite, with its size, so nothing that looks one up fails:
//! those left out sample a transparent texel. It keeps what the world draws: entities, items,
//! particles, fonts, skins, trims and a few maps.

use super::{
    GLYPH_ROWS, HudAtlas, MAP_SLOTS, MINIMAP_SPRITE, MOD_IMAGE_SIZE, MOD_IMAGE_SPRITE,
    PACK_ICON_SIZE, PACK_ICON_SLOTS, PackedImage, Rect, SKIN_SLOTS, TRIM_SLOTS, WORLD_ICON_SLOTS,
    map_slot_sprite,
};
use std::collections::HashMap;

/// What the sprites a lean atlas leaves out sample: a transparent slot.
const UNDRAWN: &str = "horizon/undrawn";
/// Framed and held maps a lean atlas draws at a time (the others stay blank).
const LEAN_MAP_SLOTS: usize = 16;
/// Its rows for glyphs rasterized at run time: signs and name tags in the world, no screens.
const LEAN_GLYPH_ROWS: u32 = 128;

/// Whether atlases are built lean (`HORIZON_LEAN_ATLAS=1`), read once.
pub(super) fn lean() -> bool {
    static LEAN: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("HORIZON_LEAN_ATLAS").is_ok_and(|v| v == "1"));
    *LEAN
}

/// Whether only screens and the HUD draw this sprite.
fn gui_only(name: &str) -> bool {
    name.starts_with("gui/") || name.contains(":gui/") || name.starts_with("horizon/jei/")
}

/// The atlas's height for sprites ending above row `glyph_region`. A multiple of 1024 rather
/// than a power of two: the default resources need 4096x5120 (8192 rows before), Faithful 32x
/// 4096x12288 (16384). A lean atlas is cut closer.
pub(super) fn height(glyph_region: u32) -> u32 {
    match lean() {
        true => (glyph_region + LEAN_GLYPH_ROWS).next_multiple_of(32),
        false => (glyph_region + GLYPH_ROWS).next_multiple_of(1024),
    }
}

/// The narrowest atlas tried. Shelves of sprites sorted by height lose up to the tallest
/// sprite's height times the width, so a lean atlas starts narrow; the full one keeps 4096.
pub(super) fn first_width() -> u32 {
    if lean() { 2048 } else { 4096 }
}

/// For a lean atlas, takes the GUI's images out of `images`: their names and sizes.
pub(super) fn take_undrawn(images: &mut Vec<PackedImage>, lean: bool) -> Vec<(String, u32, u32)> {
    let mut undrawn = Vec::new();
    if lean {
        images.retain_mut(|(name, w, h, _)| {
            let out = gui_only(name);
            if out {
                undrawn.push((std::mem::take(name), *w, *h));
            }
            !out
        });
    }
    undrawn
}

/// Names the sprites [`take_undrawn`] took, each with its size, on the transparent slot.
pub(super) fn name_undrawn(
    sprites: &mut HashMap<String, Rect>,
    undrawn: Vec<(String, u32, u32)>,
    gui_sizes: &HashMap<String, (u32, u32)>,
) {
    let Some(blank) = sprites.get(UNDRAWN).copied() else {
        return;
    };
    for (name, w, h) in undrawn {
        let (w, h) = gui_sizes.get(&name).copied().unwrap_or((w, h));
        sprites.insert(name, Rect { w, h, ..blank });
    }
}

impl HudAtlas {
    /// Empty slots written at run time: map textures (game/maps.rs), the
    /// minimap, a native mod's own image, downloaded skins, world and pack
    /// icons and the on-screen keyboard's text. Their pixels are already zero
    /// in the destination atlas, so keep the source buffers empty and avoid
    /// duplicating the atlas-sized allocation during packing.
    ///
    /// A lean atlas has the world's only: a few maps, the skins and the trims. Who asks for
    /// another finds no sprite, as each writer allows for.
    pub(super) fn slots(images: &mut Vec<PackedImage>, trim: (u32, u32), mod_image: bool) {
        let lean = lean();
        for i in 0..if lean { LEAN_MAP_SLOTS } else { MAP_SLOTS } {
            let name = map_slot_sprite(i).unwrap_or_default();
            images.push((name.into(), 128, 128, Vec::new()));
        }
        // Downloaded skins (game/scoreboard.rs).
        for i in 0..SKIN_SLOTS {
            images.push((format!("horizon/skin{i}"), 64, 64, Vec::new()));
        }
        for i in 0..TRIM_SLOTS {
            images.push((format!("horizon/trim{i}"), trim.0, trim.1, Vec::new()));
        }
        if lean {
            images.push((UNDRAWN.into(), 4, 4, Vec::new()));
            return;
        }
        // The minimap extra's image (game/extras.rs).
        images.push((
            MINIMAP_SPRITE.into(),
            crate::minimap::SIZE as u32,
            crate::minimap::SIZE as u32,
            Vec::new(),
        ));
        // A mod's own image (native `mod_image`), only when a mod is loaded.
        if mod_image {
            images.push((
                MOD_IMAGE_SPRITE.into(),
                MOD_IMAGE_SIZE,
                MOD_IMAGE_SIZE,
                Vec::new(),
            ));
        }
        for i in 0..WORLD_ICON_SLOTS {
            images.push((format!("horizon/world-icon{i}"), 64, 64, Vec::new()));
        }
        for i in 0..PACK_ICON_SLOTS {
            images.push((
                format!("horizon/pack-icon{i}"),
                PACK_ICON_SIZE,
                PACK_ICON_SIZE,
                Vec::new(),
            ));
        }
        // The on-screen keyboard's text and corner masks (ui_theme), rasterized at run time.
        images.push(("horizon/ui-text".into(), 2048, 1024, Vec::new()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lean_atlas_names_the_guis_sprites_without_their_pixels() {
        let image = |name: &str, w, h| (name.to_owned(), w, h, vec![9; (w * h * 4) as usize]);
        let mut images = vec![
            image("gui/sprites/hud/heart/full", 9, 9),
            image("entity/player/wide/steve", 64, 64),
            image("mymod:gui/panel", 200, 100),
            image("item/diamond_sword", 16, 16),
        ];
        assert!(take_undrawn(&mut images.clone(), false).is_empty());
        let undrawn = take_undrawn(&mut images, true);
        let kept: Vec<&str> = images.iter().map(|i| i.0.as_str()).collect();
        assert_eq!(kept, ["entity/player/wide/steve", "item/diamond_sword"]);

        let blank = Rect {
            u0: 0.25,
            v0: 0.5,
            u1: 0.26,
            v1: 0.51,
            w: 4,
            h: 4,
        };
        let mut sprites = HashMap::from([(UNDRAWN.to_owned(), blank)]);
        // A pack's larger texture keeps the default one's size on screen.
        let gui_sizes = HashMap::from([("mymod:gui/panel".to_owned(), (100, 50))]);
        name_undrawn(&mut sprites, undrawn, &gui_sizes);
        let heart = sprites["gui/sprites/hud/heart/full"];
        assert_eq!((heart.w, heart.h), (9, 9));
        assert_eq!((heart.u0, heart.v1), (blank.u0, blank.v1));
        let panel = sprites["mymod:gui/panel"];
        assert_eq!((panel.w, panel.h), (100, 50));
        assert!(!sprites.contains_key("item/diamond_sword"));
    }
}
