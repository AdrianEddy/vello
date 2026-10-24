// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Shared helpers for reading the packed alpha values of strips and rect strips.
//!
//! Alpha textures store 16 1-byte alpha values per texel, with each color
//! channel packing the 4 alpha values of a single strip column.

fn alpha_to_f32(alpha: u32) -> f32 {
    return f32(alpha) * (1.0 / 255.0);
}

fn sample_strip_alpha(
    alphas_texture: texture_2d<u32>,
    tex_width_bits: u32,
    tex_coord: vec2<f32>,
) -> f32 {
    let x = u32(floor(tex_coord.x));
    let y = u32(floor(tex_coord.y));
    // Retrieve alpha value from the texture. We store 16 1-byte alpha
    // values per texel, with each color channel packing 4 alpha values.
    // The code here assumes the strip height is 4, i.e., each color
    // channel encodes the alpha values for a single column within a strip.
    // Divide x by 4 to get the texel position.
    let alphas_index = x;
    let alphas_tex_width = 1u << tex_width_bits;
    // Which texel contains the alpha values for this column
    let texel_index = alphas_index / 4u;
    // Which channel (R,G,B,A) in the texel contains the alpha values for this column
    let channel_index = alphas_index % 4u;
    // Calculate texel coordinates
    let tex_x = texel_index & (alphas_tex_width - 1u);
    let tex_y = texel_index >> tex_width_bits;

    // Load all 4 channels from the texture
    let rgba_values = textureLoad(alphas_texture, vec2<u32>(tex_x, tex_y), 0);

    // Get the column's alphas from the appropriate RGBA channel based on the index
    let alphas_u32 = unpack_alphas_from_channel(rgba_values, channel_index);
    // Extract the alpha value for the current y-position from the packed u32 data
    return alpha_to_f32((alphas_u32 >> (y * 8u)) & 0xffu);
}

fn unpack_alphas_from_channel(rgba: vec4<u32>, channel_index: u32) -> u32 {
    switch channel_index {
        case 0u { return rgba.x; }
        case 1u { return rgba.y; }
        case 2u { return rgba.z; }
        case 3u { return rgba.w; }
        // Fallback, should never happen
        default { return rgba.x; }
    }
}

// Returns the alpha of `pixel` in a rect strip of `size` pixels.
//
// `alphas` holds the alphas of the first column, last column, first row and last row, from the
// low byte up. The alpha of a corner pixel is the rounded product of its column and row alpha plus
// a correction of -1, 0 or 1, which `corner_corrections` holds as 2-bit two's complement values for
// the top-left, top-right, bottom-left and bottom-right pixel, from the low bits up.
fn rect_alpha(size: vec2<u32>, pixel: vec2<u32>, alphas: u32, corner_corrections: u32) -> u32 {
    let first = pixel == vec2(0u);
    let last = pixel == size - 1u;
    var col_alpha = 255u;
    if first.x {
        col_alpha = alphas & 0xffu;
    } else if last.x {
        col_alpha = (alphas >> 8u) & 0xffu;
    }
    var row_alpha = 255u;
    if first.y {
        row_alpha = (alphas >> 16u) & 0xffu;
    } else if last.y {
        row_alpha = alphas >> 24u;
    }
    // round(col_alpha * row_alpha / 255)
    let p = col_alpha * row_alpha + 128u;
    var alpha = (p + (p >> 8u)) >> 8u;
    if (first.x || last.x) && (first.y || last.y) {
        let corner = select(1u, 0u, first.x) + select(2u, 0u, first.y);
        let correction = (corner_corrections >> (2u * corner)) & 3u;
        if correction == 1u {
            alpha += 1u;
        } else if correction == 3u {
            alpha -= 1u;
        }
    }
    return alpha;
}
