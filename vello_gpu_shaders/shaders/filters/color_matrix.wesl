// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Color matrix filter pass.

// The matrix applies to the unpremultiplied color: each `row_*` holds the weights of the input
// channels for one output channel, and `offsets` the offset of each output channel.
fn color_matrix(
    color: vec4<f32>,
    row_r: vec4<u32>,
    row_g: vec4<u32>,
    row_b: vec4<u32>,
    row_a: vec4<u32>,
    offsets: vec4<u32>,
) -> vec4<f32> {
    var straight = vec4<f32>(0.0);
    if color.a > 0.0 {
        straight = vec4<f32>(color.rgb / color.a, color.a);
    }

    let transformed = vec4<f32>(
            dot(bitcast<vec4<f32>>(row_r), straight),
            dot(bitcast<vec4<f32>>(row_g), straight),
            dot(bitcast<vec4<f32>>(row_b), straight),
            dot(bitcast<vec4<f32>>(row_a), straight),
        ) + bitcast<vec4<f32>>(offsets);
    let result = clamp(transformed, vec4<f32>(0.0), vec4<f32>(1.0));

    return vec4<f32>(result.rgb * result.a, result.a);
}
