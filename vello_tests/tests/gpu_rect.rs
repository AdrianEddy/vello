// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Coverage parity between the two ways `vello_gpu` draws axis-aligned rectangles.
//!
//! Rectangles are drawn as quads unless a clip path is active, in which case they go through the
//! strip renderer of `vello_common::rect`, like on `vello_cpu`. Both must give every pixel the same
//! coverage, so a clip path that cuts nothing must not change a byte. Snapshot tests allow a GPU
//! tolerance of one, so these tests compare two GPU renders directly. They only use solid paints.

// On wasm32, only the WebGL harness can run these.
#![cfg(any(not(target_arch = "wasm32"), feature = "webgl"))]

use vello_common::kurbo::{Rect, Shape};
use vello_common::peniko::Color;
use vello_common::pixmap::Pixmap;
use vello_cpu::RenderMode;

use crate::renderer::{GpuRenderer, Renderer};
use crate::util::get_ctx_with_depth_buffer;

/// The color `render` fills the viewport with before drawing.
const BACKGROUND: Color = Color::from_rgb8(22, 27, 34);

fn viewport(width: u16, height: u16) -> Rect {
    Rect::new(0.0, 0.0, f64::from(width), f64::from(height))
}

fn render(
    width: u16,
    height: u16,
    use_depth_buffer: bool,
    draw: impl FnOnce(&mut GpuRenderer),
) -> Pixmap {
    let mut ctx = get_ctx_with_depth_buffer::<GpuRenderer>(
        width,
        height,
        false,
        0,
        "baseline",
        RenderMode::OptimizeSpeed,
        use_depth_buffer,
    );
    ctx.set_paint(BACKGROUND);
    ctx.fill_rect(&viewport(width, height));
    draw(&mut ctx);
    ctx.render();
    ctx.snapshot()
}

fn fill_rects(ctx: &mut GpuRenderer, rects: &[(Rect, Color)]) {
    for (rect, color) in rects {
        ctx.set_paint(*color);
        ctx.fill_rect(rect);
    }
}

fn assert_same_pixels(expected: &Pixmap, actual: &Pixmap, what: &str) {
    assert_eq!(
        (expected.width(), expected.height()),
        (actual.width(), actual.height()),
        "{what}: the renders differ in size"
    );
    let width = usize::from(expected.width());
    let expected: Vec<_> = expected.data_as_u8_slice().chunks_exact(4).collect();
    let actual: Vec<_> = actual.data_as_u8_slice().chunks_exact(4).collect();
    // Two renders of only the background would match too.
    let background = BACKGROUND.premultiply().to_rgba8().to_u8_array();
    assert!(
        expected.iter().any(|pixel| *pixel != background),
        "{what}: nothing is drawn over the background"
    );
    let differing = expected.iter().zip(&actual).filter(|(e, a)| e != a).count();
    if let Some(i) = expected.iter().zip(&actual).position(|(e, a)| e != a) {
        panic!(
            "{what}: {differing} pixels differ, the first at ({}, {}): expected {:?}, got {:?}",
            i % width,
            i / width,
            expected[i],
            actual[i],
        );
    }
}

/// Render `rects` as quads and, inside a clip path covering the viewport, as strips, with and
/// without the depth buffer, optionally inside a translucent layer, and require the same bytes.
fn assert_quads_match_strips(width: u16, height: u16, in_layer: bool, rects: &[(Rect, Color)]) {
    for use_depth_buffer in [true, false] {
        let draw = |ctx: &mut GpuRenderer, clip_path: bool| {
            if in_layer {
                ctx.push_opacity_layer(0.8);
            }
            if clip_path {
                ctx.push_clip_path(&viewport(width, height).to_path(0.1));
            }
            fill_rects(ctx, rects);
            if clip_path {
                ctx.pop_clip();
            }
            if in_layer {
                ctx.pop_layer();
            }
        };
        let quads = render(width, height, use_depth_buffer, |ctx| draw(ctx, false));
        let strips = render(width, height, use_depth_buffer, |ctx| draw(ctx, true));
        assert_same_pixels(
            &strips,
            &quads,
            &format!("quads vs strips, depth buffer {use_depth_buffer}, layer {in_layer}"),
        );
    }
}

/// Rectangles with fractional edges starting at `x`, filling a band `width` by `height`: slivers,
/// rectangles with both edges in one pixel, edges less than half an alpha step from the pixel grid
/// and rectangles large enough to be split into five parts, opaque and translucent.
fn fractional_rects(x: f64, width: u16, height: u16) -> Vec<(Rect, Color)> {
    let fract = |i: u32, step: f64| (f64::from(i) * step).fract();
    let color = |i: u32| {
        let alpha = if i.is_multiple_of(3) { 0.6 } else { 1.0 };
        Color::new([
            fract(i, 0.31) as f32,
            fract(i, 0.53) as f32,
            fract(i, 0.71) as f32,
            alpha,
        ])
    };

    let mut rects = Vec::new();
    let mut i = 0;
    // A grid of small rectangles at golden-ratio offsets.
    for cell_y in 0..(height / 16 - 3) {
        for cell_x in 0..(width / 16) {
            i += 1;
            let x0 = x + f64::from(cell_x * 16) + 1.0 + 2.0 * fract(i, 0.618_033_988_75);
            let y0 = f64::from(cell_y * 16) + 1.0 + 2.0 * fract(i, 0.414_213_562_37);
            let w = 0.2 + 11.5 * fract(i, 0.732_050_807_57);
            let h = 0.2 + 11.5 * fract(i, 0.236_067_977_5);
            rects.push((Rect::new(x0, y0, x0 + w, y0 + h), color(i)));
        }
    }
    let y = f64::from(height - 48);
    for (j, rect) in [
        // Both horizontal edges in one pixel.
        Rect::new(x + 2.3, y + 1.0, x + 2.7, y + 12.0),
        Rect::new(x + 6.1, y + 2.25, x + 6.85, y + 2.5),
        // The edge alphas are 255, but the top-left alpha is 254.
        Rect::new(x + 10.001, y + 1.001, x + 30.0, y + 14.0),
        Rect::new(x + 32.0, y + 0.999, x + 52.999, y + 15.0),
        // Split into five parts.
        Rect::new(x + 56.25, y + 1.6, x + 96.75, y + 38.3),
        Rect::new(x + 100.5, y + 3.0, x + 140.0, y + 46.5),
    ]
    .into_iter()
    .enumerate()
    {
        rects.push((rect, color(i + j as u32)));
    }
    rects
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn gpu_rect_clip_path_changes_nothing() {
    assert_quads_match_strips(256, 128, false, &fractional_rects(0.0, 256, 128));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn gpu_rect_clip_path_changes_nothing_in_layer() {
    assert_quads_match_strips(256, 128, true, &fractional_rects(9.0, 240, 128));
}

/// Converting an `f64` edge to `f32` moves it by up to half an `f32` ulp, about 1.2e-4 pixels at
/// x = 4000, which is enough to change the rounding of an alpha.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn gpu_rect_clip_path_changes_nothing_at_large_coordinates() {
    assert_quads_match_strips(4400, 64, false, &fractional_rects(4000.0, 400, 64));
}

/// Under a rectangular clip, rectangles stay quads of their intersection with the clip, which
/// must look like the strips of that intersection.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn gpu_rect_clip_rect_matches_strips() {
    let (width, height) = (256, 128);
    let clip = Rect::new(13.37, 6.001, 209.5, 115.8);
    let rects = fractional_rects(0.0, width, height);
    for use_depth_buffer in [true, false] {
        let quads = render(width, height, use_depth_buffer, |ctx| {
            ctx.push_clip_rect(&clip);
            fill_rects(ctx, &rects);
            ctx.pop_clip();
        });
        let strips = render(width, height, use_depth_buffer, |ctx| {
            ctx.push_clip_path(&viewport(width, height).to_path(0.1));
            for (rect, color) in &rects {
                ctx.set_paint(*color);
                ctx.fill_rect(&rect.intersect(clip));
            }
            ctx.pop_clip();
        });
        assert_same_pixels(
            &strips,
            &quads,
            &format!("clip rect vs strips, depth buffer {use_depth_buffer}"),
        );
    }
}
