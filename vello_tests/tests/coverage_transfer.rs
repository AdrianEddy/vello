// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for coverage transfers on glyph runs, comparing glyphs drawn from the atlas and from
//! their outlines on each pipeline.

use crate::renderer::{CpuRenderer, GpuRenderer, Renderer};
use crate::util::layout_glyphs_roboto;
use glifo::{CoverageTransfer, Glyph};
use vello_common::color::palette::css::WHITE;
use vello_common::kurbo::{Affine, Circle, Shape};
use vello_common::pixmap::Pixmap;
use vello_cpu::{Level, RenderMode};

const WIDTH: u16 = 290;
const HEIGHT: u16 = 84;
const FONT_SIZE: f32 = 14.0;

/// Sharpness and weight in steps of 1/256.
type Parameters = (i16, i16);

const SHARPEN: Parameters = (154, 0);
const SOFTEN_AND_THICKEN: Parameters = (-96, 199);

fn transfer((sharpness, weight): Parameters) -> CoverageTransfer {
    CoverageTransfer::new(f32::from(sharpness) / 256.0, f32::from(weight) / 256.0)
}

/// The documented curve, evaluated in `f64`.
fn reference((sharpness, weight): Parameters, coverage: u8) -> u8 {
    let s = f64::from(sharpness) / 256.0;
    let w = f64::from(weight) / 256.0;
    let a = f64::from(coverage) / 255.0;
    let t = a + a * (1.0 - a) * (w + s * (2.0 * a - 1.0));
    ((t * 255.0).clamp(0.0, 255.0) + 0.5) as u8
}

#[derive(Clone, Copy, Debug)]
struct Draw {
    atlas_cache: bool,
    hint: bool,
    /// Whether glyphs sit at integer positions, rather than at all quarter-pixel offsets.
    integer_positions: bool,
    clip: bool,
}

/// Draw white glyphs onto a transparent target, each in its own cell so that no pixel is
/// covered by two glyphs. The output alpha is then the glyph coverage.
fn render<R: Renderer>(
    num_threads: u16,
    render_mode: RenderMode,
    draw: Draw,
    transfer: CoverageTransfer,
) -> Pixmap {
    let (font, glyphs) = layout_glyphs_roboto("Hamburgefonstiv", FONT_SIZE);
    let level = Level::try_detect().unwrap_or(Level::baseline());
    let mut ctx = R::new(WIDTH, HEIGHT, num_threads, level, render_mode);
    if draw.clip {
        ctx.push_clip_path(&Circle::new((143.3, 41.6), 38.4).to_path(0.1));
    }
    ctx.set_paint(WHITE);
    for row in 0..4_u8 {
        let x_offset = if draw.integer_positions {
            0.0
        } else {
            f32::from(row) / 4.0
        };
        ctx.set_transform(Affine::translate((4.0, 16.0 + 20.0 * f64::from(row))));
        ctx.glyph_run(&font)
            .font_size(FONT_SIZE)
            .hint(draw.hint)
            .atlas_cache(draw.atlas_cache)
            .coverage_transfer(transfer)
            .fill_glyphs(glyphs.iter().enumerate().map(|(i, glyph)| Glyph {
                x: 18.0 * i as f32 + x_offset,
                y: 0.0,
                ..*glyph
            }))
            .unwrap();
    }
    if draw.clip {
        ctx.pop_clip();
    }
    ctx.flush();
    ctx.render();
    ctx.snapshot()
}

fn alphas(pixmap: &Pixmap) -> Vec<u8> {
    pixmap.data().iter().map(|pixel| pixel.a).collect()
}

fn max_diff(a: &[u8], b: &[u8]) -> u8 {
    a.iter().zip(b).map(|(a, b)| a.abs_diff(*b)).max().unwrap()
}

/// Every pixel's coverage must be the transferred coverage of the same pixel drawn without a
/// transfer, both for glyphs drawn from their outlines and from the atlas.
fn check_transferred_coverage<R: Renderer>(num_threads: u16, render_mode: RenderMode) {
    for atlas_cache in [false, true] {
        for hint in [false, true] {
            let draw = Draw {
                atlas_cache,
                hint,
                integer_positions: false,
                clip: false,
            };
            let identity = alphas(&render::<R>(
                num_threads,
                render_mode,
                draw,
                CoverageTransfer::IDENTITY,
            ));
            for parameters in [SHARPEN, SOFTEN_AND_THICKEN] {
                let transferred = alphas(&render::<R>(
                    num_threads,
                    render_mode,
                    draw,
                    transfer(parameters),
                ));
                let expected: Vec<u8> =
                    identity.iter().map(|&a| reference(parameters, a)).collect();
                assert!(
                    max_diff(&expected, &identity) > 8,
                    "{draw:?}: the transfer must be visible"
                );
                let mismatches = transferred
                    .iter()
                    .zip(&expected)
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(mismatches, 0, "{draw:?}, {parameters:?}");
            }
        }
    }
}

#[test]
fn coverage_transfer_cpu_u8() {
    check_transferred_coverage::<CpuRenderer>(0, RenderMode::OptimizeSpeed);
}

#[test]
fn coverage_transfer_cpu_f32() {
    check_transferred_coverage::<CpuRenderer>(0, RenderMode::OptimizeQuality);
}

#[test]
fn coverage_transfer_cpu_multithreaded_u8() {
    check_transferred_coverage::<CpuRenderer>(3, RenderMode::OptimizeSpeed);
}

#[test]
fn coverage_transfer_cpu_multithreaded_f32() {
    check_transferred_coverage::<CpuRenderer>(3, RenderMode::OptimizeQuality);
}

#[test]
fn coverage_transfer_gpu() {
    check_transferred_coverage::<GpuRenderer>(0, RenderMode::OptimizeSpeed);
}

/// Under an anti-aliased clip, glyphs drawn from the atlas and from their outlines get the
/// transfer before they are clipped.
fn check_clipped_cached_matches_direct<R: Renderer>(render_mode: RenderMode) {
    let draw = |atlas_cache| Draw {
        atlas_cache,
        hint: true,
        integer_positions: true,
        clip: true,
    };
    let transfer = transfer(SHARPEN);
    let direct = alphas(&render::<R>(0, render_mode, draw(false), transfer));
    let cached = alphas(&render::<R>(0, render_mode, draw(true), transfer));
    let direct_identity = alphas(&render::<R>(
        0,
        render_mode,
        draw(false),
        CoverageTransfer::IDENTITY,
    ));
    let cached_identity = alphas(&render::<R>(
        0,
        render_mode,
        draw(true),
        CoverageTransfer::IDENTITY,
    ));

    assert!(
        max_diff(&direct, &direct_identity) > 8,
        "the transfer must apply to glyphs drawn from their outlines"
    );
    assert!(
        max_diff(&cached, &cached_identity) > 8,
        "the transfer must apply to glyphs drawn from the atlas"
    );
    let diff = max_diff(&cached, &direct);
    assert!(diff <= 1, "cached and direct glyphs differ by {diff}");
}

#[test]
fn coverage_transfer_clipped_cached_matches_direct_cpu_u8() {
    check_clipped_cached_matches_direct::<CpuRenderer>(RenderMode::OptimizeSpeed);
}

#[test]
fn coverage_transfer_clipped_cached_matches_direct_cpu_f32() {
    check_clipped_cached_matches_direct::<CpuRenderer>(RenderMode::OptimizeQuality);
}

#[test]
fn coverage_transfer_clipped_cached_matches_direct_gpu() {
    check_clipped_cached_matches_direct::<GpuRenderer>(RenderMode::OptimizeSpeed);
}
