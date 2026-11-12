// The offscreen picture as luma bytes (docs/rl/design.md, "Images"): one invocation packs four
// horizontally adjacent pixels into one u32 of the output, in row order, so the bytes are the
// (height, width) picture with no row padding.
//
// The input is the frame as the display shows it (8-bit, the 2.2 gamma already applied), copied
// into a buffer with rows `row_words` u32s apart. Luma is Rec. 601's weights on those gamma
// values (Y'), the usual grayscale of RL pipelines.

struct Params {
    width: u32,
    height: u32,
    row_words: u32,
    // 1: the bytes are B, G, R, A (Bgra8Unorm); 0: R, G, B, A.
    bgra: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> frame: array<u32>;
@group(0) @binding(2) var<storage, read_write> luma: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let words = params.width / 4u;
    if id.x >= words || id.y >= params.height {
        return;
    }
    var packed = 0u;
    for (var i = 0u; i < 4u; i++) {
        let c = unpack4x8unorm(frame[id.y * params.row_words + id.x * 4u + i]);
        var rgb = c.xyz;
        if params.bgra == 1u {
            rgb = c.zyx;
        }
        let y = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
        packed |= u32(round(saturate(y) * 255.0)) << (8u * i);
    }
    luma[id.y * words + id.x] = packed;
}
