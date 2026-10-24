// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for glyph rendering.

use crate::renderer::Renderer;
#[cfg(target_os = "macos")]
use crate::util::layout_glyphs_apple_color_emoji;
use crate::util::{
    layout_glyphs_noto_cbtf, layout_glyphs_noto_colr, layout_glyphs_roboto,
    stops_blue_green_red_yellow,
};
use glifo::{CoverageTransfer, FontEmbolden, Glyph};
use std::f64::consts::FRAC_PI_4;
use std::iter;
use std::sync::Arc;
use vello_common::color::Srgb;
use vello_common::color::palette::css::{BLACK, BLUE, GREEN, REBECCA_PURPLE, WHITE};
use vello_common::kurbo::{Affine, Diagonal2, Point, Rect, Stroke};
use vello_common::paint::{Image, PaintType, PremulColor};
use vello_common::peniko::{
    Blob, Color, Extend, FontData, Gradient, ImageQuality, ImageSampler, LinearGradientPosition,
};
use vello_common::pixmap::{PixelMetadata, Pixmap};
use vello_dev_macros::vello_test;
use vello_gpu::GlyphMaintenance;

fn render_transform_composition_rows(
    ctx: &mut impl Renderer,
    enable_caching: bool,
    hint: bool,
    reverse_x_shift: f64,
    paint: impl Into<PaintType>,
    layout: impl Fn(f32) -> (FontData, Vec<Glyph>),
) {
    let rows = [
        (Affine::IDENTITY, 20.0_f32, Affine::IDENTITY),
        (Affine::scale(20.0), 1.0_f32, Affine::IDENTITY),
        (Affine::IDENTITY, 10.0_f32, Affine::scale(2.0)),
        (Affine::scale(2.0), 5.0_f32, Affine::scale(2.0)),
        (
            Affine::translate((-4.0, 0.0)),
            20.0_f32,
            Affine::translate((4.0, 0.0)),
        ),
        (
            Affine::translate((-4.0, 0.0)) * Affine::scale(4.0),
            5.0_f32,
            Affine::translate((1.0, 0.0)),
        ),
        (
            Affine::translate((-1.0, 0.0)),
            40.0_f32,
            Affine::scale(0.5) * Affine::translate((2.0, 0.0)),
        ),
        (
            Affine::IDENTITY,
            20.0_f32,
            Affine::translate((10.0, -10.0))
                * Affine::rotate(FRAC_PI_4)
                * Affine::translate((-10.0, 10.0)),
        ),
        (
            Affine::IDENTITY,
            20.0_f32,
            Affine::translate((10.0, -10.0))
                * Affine::skew(0.35, 0.0)
                * Affine::translate((-10.0, 10.0)),
        ),
        (
            Affine::IDENTITY,
            20.0_f32,
            Affine::translate((10.0, -10.0))
                * Affine::skew(0.0, 0.2)
                * Affine::translate((-10.0, 10.0)),
        ),
        (
            Affine::scale_non_uniform(1.0, -1.0) * Affine::translate((0.0, 20.0)),
            20.0_f32,
            Affine::IDENTITY,
        ),
        (
            Affine::scale_non_uniform(-1.0, 1.0) * Affine::translate((-reverse_x_shift, 0.0)),
            20.0_f32,
            Affine::IDENTITY,
        ),
        (
            Affine::scale_non_uniform(-1.0, -1.0) * Affine::translate((-reverse_x_shift, 20.0)),
            20.0_f32,
            Affine::IDENTITY,
        ),
    ];

    ctx.set_paint(paint);

    let mut y = 28.35;
    for (run_scale_transform, font_size, glyph_transform) in rows {
        let (font, glyphs) = layout(font_size);
        ctx.set_transform(Affine::translate((16.0, y)) * run_scale_transform);
        ctx.glyph_run(&font)
            .font_size(font_size)
            .atlas_cache(enable_caching)
            .glyph_transform(glyph_transform)
            .hint(hint)
            .fill_glyphs(glyphs.into_iter())
            .unwrap();
        y += 30.0;
    }
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_filled(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_filled_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 250, height = 70)]
fn glyphs_invalid_id_does_not_stop_run(ctx: &mut impl Renderer) {
    let font_size = 50.0_f32;
    let (font, mut glyphs) = layout_glyphs_roboto("HelloWorld", font_size);
    let invalid_index = glyphs.len() / 2;
    glyphs[invalid_index].id = u32::MAX;

    ctx.set_transform(Affine::translate((0.0, f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    let result = ctx
        .glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(false)
        .hint(false)
        .fill_glyphs(glyphs.into_iter());

    let error = result.expect_err("invalid glyph ID should be reported");
    assert_eq!(
        error.first_skipped.index, invalid_index,
        "the invalid glyph should be reported at its position in the run"
    );
    assert_eq!(
        error.first_skipped.glyph.id,
        u32::MAX,
        "the reported glyph should be the invalid one"
    );
    assert_eq!(
        error.first_skipped.reason,
        glifo::GlyphSkipReason::NoRenderableRepresentation,
        "an invalid glyph ID has no renderable representation"
    );
    assert_eq!(
        error.skipped_count, 1,
        "only the invalid glyph should be skipped"
    );
}

#[vello_test(width = 760, height = 140)]
fn glyphs_emboldened(ctx: &mut impl Renderer) {
    let font_size: f32 = 44_f32;
    let (font, glyphs) = layout_glyphs_roboto("this is regular and emboldened text", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();

    let (font, glyphs) = layout_glyphs_roboto("this is regular and emboldened text", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size) + 58.0)));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .font_embolden(FontEmbolden::new(Diagonal2::new(1.0, 1.0)))
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

/// Dark text on a light background and light text on a dark one, each without and with a coverage
/// transfer.
#[vello_test(width = 240, height = 64, glyph)]
fn glyphs_filled_coverage_transfer(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size = 13.0_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hamburgefonstiv", font_size);

    ctx.set_paint(BLACK);
    ctx.fill_rect(&Rect::new(120.0, 0.0, 240.0, 64.0));

    let sharpen = CoverageTransfer::new(0.6, 0.0);
    let sharpen_and_thicken = CoverageTransfer::new(0.6, 0.4);
    for (x, y, color, transfer) in [
        (4.0, 24.0, BLACK, CoverageTransfer::IDENTITY),
        (4.0, 52.0, BLACK, sharpen),
        (124.0, 24.0, WHITE, CoverageTransfer::IDENTITY),
        (124.0, 52.0, WHITE, sharpen_and_thicken),
    ] {
        ctx.set_transform(Affine::translate((x, y)));
        ctx.set_paint(color);
        ctx.glyph_run(&font)
            .font_size(font_size)
            .atlas_cache(enable_caching)
            .coverage_transfer(transfer)
            .fill_glyphs(glyphs.iter().copied())
            .unwrap();
    }
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_stroked(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(true)
        .stroke_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_stroked_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .stroke_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_stroked_scaled_up(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 5_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(10.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(Stroke {
        width: 0.3,
        ..Stroke::default()
    });
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .stroke_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_large_stroke_width(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.set_stroke(Stroke {
        width: 3.0,
        ..Stroke::default()
    });
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .stroke_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 300, height = 120)]
fn glyphs_stroked_then_filled(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    render_roboto_with_mode(
        ctx,
        &font,
        font_size,
        glyphs.iter().copied(),
        Affine::translate((0., f64::from(font_size))),
        DrawMode::Stroke,
    );
    render_roboto_with_mode(
        ctx,
        &font,
        font_size,
        glyphs.into_iter(),
        Affine::translate((0., f64::from(font_size * 2.0))),
        DrawMode::Fill,
    );
}

#[vello_test(width = 300, height = 120)]
fn glyphs_filled_then_stroked(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    render_roboto_with_mode(
        ctx,
        &font,
        font_size,
        glyphs.iter().copied(),
        Affine::translate((0., f64::from(font_size))),
        DrawMode::Fill,
    );
    render_roboto_with_mode(
        ctx,
        &font,
        font_size,
        glyphs.into_iter(),
        Affine::translate((0., f64::from(font_size * 2.0))),
        DrawMode::Stroke,
    );
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_skewed(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 300, height = 70, glyph)]
fn glyphs_skewed_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(-20_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 250, height = 75, glyph)]
fn glyphs_skewed_long(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 20_f32;
    let (font, glyphs) = layout_glyphs_roboto(
        "Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\nSed ornare arcu lectus.",
        font_size,
    );

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(-10_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 250, height = 75, glyph)]
fn glyphs_skewed_long_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 20_f32;
    let (font, glyphs) = layout_glyphs_roboto(
        "Lorem ipsum dolor sit amet,\nconsectetur adipiscing elit.\nSed ornare arcu lectus.",
        font_size,
    );

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(-10_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_skewed_unskewed(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(
        Affine::translate((0., f64::from(font_size)))
            * Affine::skew(-20_f64.to_radians().tan(), 0.),
    );
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(20_f64.to_radians().tan(), 0.))
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_skewed_unskewed_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(
        Affine::translate((0., f64::from(font_size)))
            * Affine::skew(-20_f64.to_radians().tan(), 0.),
    );
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::skew(20_f64.to_radians().tan(), 0.))
        .hint(false)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_scaled(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_scaled_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_glyph_transform(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 150, height = 125, glyph)]
fn glyphs_glyph_transform_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 25_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello,\nworld!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))).then_scale(2.0));
    ctx.set_paint(REBECCA_PURPLE.with_alpha(0.5));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .glyph_transform(Affine::translate((10., 10.)))
        .hint(false)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

enum GlyphPaint {
    Gradient,
    Image,
}

fn glyphs_with_transformed_paint_inner(
    ctx: &mut impl Renderer,
    enable_caching: bool,
    mode: DrawMode,
    glyph_paint: GlyphPaint,
) {
    const SCENE_WIDTH: f64 = 300.0;
    const GRADIENT_WIDTH: f64 = SCENE_WIDTH / 2.0;
    const IMAGE_WIDTH: f64 = 4.0;
    const BASELINE_DELTA: f64 = 40.0;

    let font_size = 36.0_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello World", font_size);
    let text_width = glyphs
        .last()
        .map_or(0.0, |glyph| f64::from(glyph.x) + f64::from(font_size) * 0.6);
    let centered_x = ((SCENE_WIDTH - text_width) / 2.0) as f32;
    let paint = match glyph_paint {
        GlyphPaint::Gradient => PaintType::from(Gradient {
            kind: LinearGradientPosition {
                start: Point::new(0.0, 0.0),
                end: Point::new(GRADIENT_WIDTH, 0.0),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        }),
        GlyphPaint::Image => {
            let color_strip = stops_blue_green_red_yellow();
            let pixels = color_strip
                .0
                .iter()
                .flat_map(|stop| {
                    PremulColor::from_alpha_color(stop.color.to_alpha_color::<Srgb>())
                        .as_premul_rgba8()
                        .to_u8_array()
                })
                .collect();
            let image = ctx.get_image_source(Arc::new(Pixmap::from_parts(
                pixels,
                4,
                1,
                PixelMetadata::default(),
            )));
            PaintType::from(Image {
                image,
                sampler: ImageSampler {
                    x_extend: Extend::Pad,
                    y_extend: Extend::Pad,
                    // TODO: There seems to be a mismatch when using `Medium` here between GPU and CPU,
                    // so let's use `Low` for now.
                    quality: ImageQuality::Low,
                    alpha: 1.0,
                },
            })
        }
    };

    ctx.set_paint(paint);
    ctx.set_stroke(Stroke {
        width: 1.5,
        ..Stroke::default()
    });

    let mut baseline_y = 42.0;
    // 1) Visible gradient starts at green since we only translate the
    // glyph X, which shouldn't affect the paint transform.
    // 2) Visible gradient starts at 2, since paint is transformed along with
    // the path transform.
    // 3) Same as 2), just that the source is the paint transform instead of the
    // scene transform.
    // 4) Gradient should be shifted to the right, since it's affected by
    // scene transform + paint transform.
    for (use_render_transform, shift_gradient) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let (transform, glyph_x) = if use_render_transform {
            (Affine::translate((centered_x as f64, baseline_y)), 0.0)
        } else {
            (Affine::translate((0.0, baseline_y)), centered_x)
        };
        let paint_transform = match glyph_paint {
            GlyphPaint::Gradient => {
                if shift_gradient {
                    Affine::translate((centered_x as f64, 0.0))
                } else {
                    Affine::IDENTITY
                }
            }
            GlyphPaint::Image => {
                let image_transform = Affine::scale_non_uniform(GRADIENT_WIDTH / IMAGE_WIDTH, 1.0);
                if shift_gradient {
                    Affine::translate((centered_x as f64, 0.0)) * image_transform
                } else {
                    image_transform
                }
            }
        };
        let row_glyphs = glyphs.iter().map(|glyph| Glyph {
            id: glyph.id,
            x: glyph.x + glyph_x,
            y: glyph.y,
        });

        ctx.set_transform(transform);
        ctx.set_paint_transform(paint_transform);
        let builder = ctx
            .glyph_run(&font)
            .font_size(font_size)
            .atlas_cache(enable_caching)
            .hint(false);

        match mode {
            DrawMode::Fill => builder.fill_glyphs(row_glyphs),
            DrawMode::Stroke => builder.stroke_glyphs(row_glyphs),
        }
        .unwrap();

        baseline_y += BASELINE_DELTA;
    }
}

#[vello_test(width = 300, height = 180, glyph)]
fn glyphs_with_gradient(ctx: &mut impl Renderer, enable_caching: bool) {
    glyphs_with_transformed_paint_inner(ctx, enable_caching, DrawMode::Fill, GlyphPaint::Gradient);
}

#[vello_test(width = 300, height = 180, glyph)]
fn glyphs_with_gradient_stroked(ctx: &mut impl Renderer, enable_caching: bool) {
    glyphs_with_transformed_paint_inner(
        ctx,
        enable_caching,
        DrawMode::Stroke,
        GlyphPaint::Gradient,
    );
}

#[vello_test(width = 300, height = 180, glyph)]
fn glyphs_with_image(ctx: &mut impl Renderer, enable_caching: bool) {
    glyphs_with_transformed_paint_inner(ctx, enable_caching, DrawMode::Fill, GlyphPaint::Image);
}

#[vello_test(width = 110, height = 410, glyph, gpu_tolerance = 1)]
fn glyphs_transform_composition_rows_outline(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(
        ctx,
        enable_caching,
        false,
        47.0,
        REBECCA_PURPLE.with_alpha(0.5),
        |font_size| layout_glyphs_roboto("Hello", font_size),
    );
}

#[vello_test(width = 110, height = 410, glyph, gpu_tolerance = 1)]
fn glyphs_transform_composition_rows_outline_hinted(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(
        ctx,
        enable_caching,
        true,
        47.0,
        REBECCA_PURPLE.with_alpha(0.5),
        |font_size| layout_glyphs_roboto("Hello", font_size),
    );
}

// Next two tests require high tolerance on CPU likely due to having to use bicubic interpolation, since
// we downscale a lot.

#[vello_test(width = 210, height = 410, skip_gpu, glyph, cpu_u8_tolerance = 3)]
fn glyphs_transform_composition_rows_bitmap(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(ctx, enable_caching, false, 100.0, BLACK, |font_size| {
        layout_glyphs_noto_cbtf("✅👀🎉🤠", font_size)
    });
}

#[vello_test(width = 210, height = 410, skip_gpu, glyph, cpu_u8_tolerance = 3)]
fn glyphs_transform_composition_rows_bitmap_hinted(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(ctx, enable_caching, true, 100.0, BLACK, |font_size| {
        layout_glyphs_noto_cbtf("✅👀🎉🤠", font_size)
    });
}

// TODO: The cached versions of COLR glyphs seem to have a slight shift, investigate.

#[vello_test(
    width = 210,
    height = 410,
    cpu_u8_tolerance = 3,
    gpu_tolerance = 3,
    glyph
)]
fn glyphs_transform_composition_rows_colr(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(ctx, enable_caching, false, 100.0, BLACK, |font_size| {
        layout_glyphs_noto_colr("✅👀🎉🤠", font_size)
    });
}

#[vello_test(
    width = 210,
    height = 410,
    cpu_u8_tolerance = 3,
    gpu_tolerance = 3,
    glyph
)]
fn glyphs_transform_composition_rows_colr_hinted(ctx: &mut impl Renderer, enable_caching: bool) {
    render_transform_composition_rows(ctx, enable_caching, true, 100.0, BLACK, |font_size| {
        layout_glyphs_noto_colr("✅👀🎉🤠", font_size)
    });
}

#[vello_test(width = 60, height = 12, glyph)]
fn glyphs_small(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 60, height = 12, glyph)]
fn glyphs_small_unhinted(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 10_f32;
    let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.set_paint(REBECCA_PURPLE);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[vello_test(width = 250, height = 70, skip_gpu, glyph, cpu_u8_tolerance = 1)]
fn glyphs_bitmap_noto(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_noto_cbtf("✅👀🎉🤠", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrawMode {
    Fill,
    Stroke,
}

fn render_roboto_with_mode(
    ctx: &mut impl Renderer,
    font: &FontData,
    font_size: f32,
    glyphs: impl Iterator<Item = Glyph> + Clone,
    transform: Affine,
    mode: DrawMode,
) {
    ctx.set_transform(transform);
    ctx.set_paint(REBECCA_PURPLE);
    ctx.set_stroke(Stroke {
        width: 3.0,
        ..Stroke::default()
    });
    let builder = ctx.glyph_run(font).font_size(font_size);

    match mode {
        DrawMode::Fill => {
            builder.fill_glyphs(glyphs).unwrap();
        }
        DrawMode::Stroke => {
            builder.stroke_glyphs(glyphs).unwrap();
        }
    }
}

#[vello_test(
    width = 250,
    height = 70,
    cpu_u8_tolerance = 1,
    gpu_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 250,
    height = 70,
    cpu_u8_tolerance = 1,
    gpu_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_stroked(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(width = 100, height = 100, glyph)]
fn glyphs_colr_noto_overflow_centered(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size = 150.0;
    let (font, mut glyphs) = layout_glyphs_noto_colr("✅", font_size);
    let glyph = glyphs.pop().unwrap();

    let centered_glyph = Glyph {
        id: glyph.id,
        x: -25.0,
        y: 125.0,
    };

    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false)
        .fill_glyphs(iter::once(centered_glyph))
        .unwrap();
}

#[vello_test(
    width = 500,
    height = 140,
    cpu_u8_tolerance = 1,
    gpu_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_scaled_2x(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)).then_scale(2.0),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 125,
    height = 35,
    cpu_u8_tolerance = 1,
    gpu_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_scaled_half(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)).then_scale(0.5),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 350,
    height = 350,
    cpu_u8_tolerance = 3,
    gpu_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_rotated(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((175., 100.)) * Affine::rotate(FRAC_PI_4),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 600,
    height = 600,
    cpu_u8_tolerance = 2,
    gpu_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_rotated_scaled(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((300., 150.)) * Affine::rotate(FRAC_PI_4) * Affine::scale(2.0),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 250,
    height = 140,
    cpu_u8_tolerance = 1,
    gpu_tolerance = 2,
    glyph
)]
fn glyphs_colr_noto_scaled_non_uniform(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((0., 50.)) * Affine::scale_non_uniform(1.0, 2.0),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(
    width = 300,
    height = 300,
    cpu_u8_tolerance = 2,
    gpu_tolerance = 1,
    glyph
)]
fn glyphs_colr_noto_rotated_scaled_non_uniform(ctx: &mut impl Renderer, enable_caching: bool) {
    render_colr_noto_with_transform(
        ctx,
        Affine::translate((150., 150.))
            * Affine::rotate(FRAC_PI_4)
            * Affine::scale_non_uniform(1.0, 2.0),
        enable_caching,
        DrawMode::Fill,
    );
}

#[vello_test(width = 250, height = 70, skip_gpu, cpu_u8_tolerance = 1)]
fn glyphs_bitmap_noto_stroked(ctx: &mut impl Renderer) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_noto_cbtf("✅👀🎉🤠", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .stroke_glyphs(glyphs.into_iter())
        .unwrap();
}

#[cfg(target_os = "macos")]
#[vello_test(width = 200, height = 70, skip_gpu, cpu_u8_tolerance = 2, glyph)]
fn glyphs_bitmap_apple(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_apple_color_emoji("✅👀🎉🤠", font_size);

    ctx.set_transform(Affine::translate((0., f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
}

// Note that there are still four cases (which can be reduced to 2 underlying issues)
// that don't yet render the same as in CoreText.
#[vello_test(width = 288, height = 240, skip_gpu, glyph)]
fn glyphs_bitmap_sbix_glyf(ctx: &mut impl Renderer, enable_caching: bool) {
    draw_sbix_grid(
        ctx,
        enable_caching,
        include_bytes!("../../assets/sbix/sbix.ttf"),
    );
}

// Couple of glyphs are misplaced, due to the same bug affecting two glyphs
// in the `glyf` test.
#[vello_test(width = 288, height = 240, skip_gpu, glyph)]
fn glyphs_bitmap_sbix_cff(ctx: &mut impl Renderer, enable_caching: bool) {
    draw_sbix_grid(
        ctx,
        enable_caching,
        include_bytes!("../../assets/sbix/sbix.otf"),
    );
}

fn draw_sbix_grid(ctx: &mut impl Renderer, enable_caching: bool, font_data: &'static [u8]) {
    let font = FontData::new(Blob::new(Arc::new(font_data)), 0);

    ctx.set_paint(WHITE);
    ctx.fill_rect(&Rect::new(0.0, 0.0, 288.0, 240.0));
    ctx.set_paint(Color::from_rgb8(219, 226, 237));
    for column in 0..=6 {
        let x = (f64::from(column) * 48.0).min(287.0);
        ctx.fill_rect(&Rect::new(x, 0.0, x + 1.0, 240.0));
    }
    for row in 0..=5 {
        let y = (f64::from(row) * 48.0).min(239.0);
        ctx.fill_rect(&Rect::new(0.0, y, 288.0, y + 1.0));
    }

    let glyphs = (0..28).map(|i| Glyph {
        id: i + 2,
        x: (i % 6) as f32 * 48.0 - 34.0,
        y: (i / 6) as f32 * 48.0 + 106.0,
    });
    ctx.set_paint(BLACK);
    let _ = ctx
        .glyph_run(&font)
        .font_size(64.0)
        .hint(false)
        .atlas_cache(enable_caching)
        .fill_glyphs(glyphs);
}

// In case anything changes here, compare to https://chromium.googlesource.com/chromium/src/+/main/third_party/blink/web_tests/platform/linux/virtual/text-antialias/colrv1-expected.png
#[vello_test(width = 400, height = 960, gpu_tolerance = 1, diff_pixels = 55, glyph)]
fn glyphs_colr_test_glyphs(ctx: &mut impl Renderer, enable_caching: bool) {
    const TEST_FONT: &[u8] =
        include_bytes!("../../assets/colr_test_glyphs/test_glyphs-glyf_colr_1.ttf");
    let font = FontData::new(Blob::new(Arc::new(TEST_FONT)), 0);
    let num_glyphs = 221;

    let font_size = 40_f64;
    let cols = 10.0;

    let width = (font_size * cols) as u16;

    ctx.set_paint(BLACK);

    let mut cur_x = 0.0;
    let mut cur_y = font_size;

    let draw_glyphs = (0..num_glyphs).filter(|n| match n {
        // Those are not COLR glyphs.
        0..8 => false,
        161..=165 => false,
        170..=176 => false,
        _ => true,
    });

    for id in draw_glyphs {
        if cur_x >= width as f64 {
            cur_x = 0.0;
            cur_y += font_size;
        }

        let glyph_iter = iter::once(Glyph { id, x: 0.0, y: 0.0 });

        ctx.set_transform(Affine::translate((cur_x, cur_y)));
        ctx.glyph_run(&font)
            .font_size(font_size as f32)
            .atlas_cache(enable_caching)
            .fill_glyphs(glyph_iter)
            .unwrap();

        cur_x += font_size;
    }

    cur_y += font_size;

    for color in [BLUE, GREEN] {
        cur_x = 0.0;
        cur_y += font_size;

        ctx.set_paint(color);

        for g in 148..=153 {
            let glyph_iter = iter::once(Glyph {
                id: g,
                x: 0.0,
                y: 0.0,
            });

            ctx.set_transform(Affine::translate((cur_x, cur_y)));
            ctx.glyph_run(&font)
                .font_size(font_size as f32)
                .atlas_cache(enable_caching)
                .fill_glyphs(glyph_iter)
                .unwrap();

            cur_x += font_size;
        }
    }
}

/// Hinting is disabled to preserve transforms passed to `prepare_colr_glyph`.
fn render_colr_noto_with_transform(
    ctx: &mut impl Renderer,
    transform: Affine,
    enable_caching: bool,
    mode: DrawMode,
) {
    let font_size: f32 = 50_f32;
    let (font, glyphs) = layout_glyphs_noto_colr("✅👀🎉🤠", font_size);

    ctx.set_transform(transform);
    let run = ctx
        .glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false);

    if mode == DrawMode::Stroke {
        run.stroke_glyphs(glyphs.into_iter()).unwrap();
    } else {
        run.fill_glyphs(glyphs.into_iter()).unwrap();
    }
}

fn render_decorated_text(
    ctx: &mut impl Renderer,
    text: &str,
    font_size: f32,
    enable_caching: bool,
    glyph_transform: Option<Affine>,
    offset: f32,
    size: f32,
    buffer: f32,
) {
    let (font, glyphs) = layout_glyphs_roboto(text, font_size);

    ctx.set_paint(REBECCA_PURPLE);

    let mut fill_builder = ctx
        .glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(false);
    if let Some(gt) = glyph_transform {
        fill_builder = fill_builder.glyph_transform(gt);
    }
    fill_builder.fill_glyphs(glyphs.iter().copied()).unwrap();

    let x_end = glyphs.last().map_or(0.0, |g| g.x + font_size * 0.6);
    let mut deco_builder = ctx
        .glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(enable_caching)
        .hint(glyph_transform.is_none());
    if let Some(gt) = glyph_transform {
        deco_builder = deco_builder.glyph_transform(gt);
    }
    deco_builder.render_decoration(glyphs.into_iter(), 0.0..=x_end, 0.0, offset, size, buffer);
}

#[vello_test(width = 300, height = 180, glyph)]
fn glyphs_decoration_offset_values(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size = 30.0_f32;
    for (i, offset) in [-6.0_f32, -2.0, 0.0, 8.0, 15.0].iter().enumerate() {
        let y = 30.0 + (i as f64) * 32.0;
        ctx.set_transform(Affine::translate((0., y)));
        render_decorated_text(
            ctx,
            "Happy joyful",
            font_size,
            enable_caching,
            None,
            *offset,
            1.5,
            1.5,
        );
    }
}

#[vello_test(width = 180, height = 180, glyph)]
fn glyphs_decoration_size_values(ctx: &mut impl Renderer, enable_caching: bool) {
    let font_size = 30.0_f32;
    for (i, size) in [0.5_f32, 1.0, 2.0, 4.0].iter().enumerate() {
        let y = 30.0 + (i as f64) * 38.0;
        ctx.set_transform(Affine::translate((0., y)));
        render_decorated_text(
            ctx,
            "Happy joyful",
            font_size,
            enable_caching,
            None,
            -2.0,
            *size,
            1.5,
        );
    }
}

#[vello_test(width = 180, height = 70, glyph)]
fn glyphs_decoration_no_descenders(ctx: &mut impl Renderer, enable_caching: bool) {
    ctx.set_transform(Affine::translate((0., 50.)));
    render_decorated_text(ctx, "HELLO", 50.0, enable_caching, None, -2.0, 2.0, 1.5);
}

#[vello_test(width = 100, height = 150, glyph, gpu_tolerance = 1)]
fn glyphs_decoration_transformed(ctx: &mut impl Renderer, enable_caching: bool) {
    let text = "Happy";
    let rows: [(Affine, f32, Option<Affine>, f64); 4] = [
        // Run-level scale absorbs into font_size and underline scaled.
        (Affine::scale(2.0), 12.0, None, 30.0),
        // Glyph-level scale (affects skip-ink outline transform)
        (Affine::IDENTITY, 10.0, Some(Affine::scale(1.2)), 40.0),
        // Y-flip (decoration should appear above the flipped text)
        (
            Affine::scale_non_uniform(1.0, -1.0) * Affine::translate((0.0, 20.0)),
            20.0,
            None,
            10.0,
        ),
        // Rotated text
        (Affine::rotate(FRAC_PI_4), 12.0, None, 40.0),
    ];

    let mut y = 30.0;
    for (run_transform, font_size, glyph_transform, buffer) in rows {
        ctx.set_transform(Affine::translate((16.0, y)) * run_transform);
        render_decorated_text(
            ctx,
            text,
            font_size,
            enable_caching,
            glyph_transform,
            -1.0,
            1.0,
            1.0,
        );
        y += buffer;
    }
}

fn draw_cached_text(ctx: &mut impl Renderer, text: &str) {
    let font_size = 50.0;
    let (font, glyphs) = layout_glyphs_roboto(text, font_size);
    ctx.set_transform(Affine::translate((0.0, f64::from(font_size))));
    ctx.set_paint(BLACK);
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(true)
        .hint(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
    ctx.flush();
}

/// The number of maintenance passes after which a glyph that is no longer drawn has been evicted.
fn glyph_eviction_passes() -> u64 {
    let config = glifo::GlyphCacheConfig::default();
    config.max_entry_age + config.eviction_frequency
}

fn evict_all_glyphs(ctx: &mut impl Renderer) {
    ctx.reset();
    for _ in 0..glyph_eviction_passes() {
        ctx.render();
        ctx.maintain_glyphs();
    }
}

#[vello_test(width = 300, height = 70, transparent, no_ref)]
fn glyphs_atlas_explicit_maintenance(ctx: &mut impl Renderer) {
    ctx.set_glyph_maintenance(GlyphMaintenance::Explicit);
    draw_cached_text(ctx, "Hello, world!");
    ctx.render();
    let expected = ctx.snapshot();
    assert!(
        expected.data_as_u8_slice().iter().any(|&byte| byte != 0),
        "glyphs should have been rendered"
    );

    for _ in 0..2 * glyph_eviction_passes() {
        ctx.render();
    }
    assert!(
        ctx.snapshot().data_as_u8_slice() == expected.data_as_u8_slice(),
        "rendering should not age the glyph caches"
    );

    // The glyphs are cached again, in the atlas space their eviction freed.
    evict_all_glyphs(ctx);
    draw_cached_text(ctx, "Hello, world!");
    ctx.render();
    assert!(
        ctx.snapshot().data_as_u8_slice() == expected.data_as_u8_slice(),
        "glyphs cached after an eviction should render the same"
    );
}

#[vello_test(width = 300, height = 70, transparent, no_ref)]
fn glyphs_atlas_explicit_maintenance_unrendered(ctx: &mut impl Renderer) {
    ctx.set_glyph_maintenance(GlyphMaintenance::Explicit);
    draw_cached_text(ctx, "abc");
    ctx.render();

    // Cache outline and bitmap glyphs, then drop the scene without rendering it.
    ctx.reset();
    draw_cached_text(ctx, "Hello");
    let font_size = 50.0;
    let (font, glyphs) = layout_glyphs_noto_cbtf("✅👀", font_size);
    ctx.set_transform(Affine::translate((150.0, f64::from(font_size))));
    ctx.glyph_run(&font)
        .font_size(font_size)
        .atlas_cache(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
    ctx.reset();
    for _ in 0..glyph_eviction_passes() {
        ctx.maintain_glyphs();
    }

    draw_cached_text(ctx, "world");
    ctx.render();
    let rendered = ctx.snapshot();
    evict_all_glyphs(ctx);
    draw_cached_text(ctx, "world");
    ctx.render();
    assert!(
        ctx.snapshot().data_as_u8_slice() == rendered.data_as_u8_slice(),
        "glyphs cached after an unrendered scene should render as in a clean atlas"
    );
}

#[vello_test(width = 300, height = 70, transparent, no_ref)]
fn glyphs_atlas_explicit_maintenance_image_upload(ctx: &mut impl Renderer) {
    ctx.set_glyph_maintenance(GlyphMaintenance::Explicit);
    draw_cached_text(ctx, "Hello, world!");
    ctx.render();
    evict_all_glyphs(ctx);

    // On the GPU, the image is uploaded into the atlas space freed by the evicted glyphs.
    let color = [0, 0, 255, 255];
    let mut pixmap = Pixmap::new(300, 70);
    for pixel in pixmap.data_as_u8_slice_mut().chunks_exact_mut(4) {
        pixel.copy_from_slice(&color);
    }
    let image = ctx.get_image_source(Arc::new(pixmap));
    ctx.set_transform(Affine::IDENTITY);
    ctx.set_paint(Image {
        image,
        sampler: ImageSampler::default(),
    });
    ctx.fill_rect(&Rect::new(0.0, 0.0, 300.0, 70.0));
    ctx.flush();
    ctx.render();
    assert!(
        ctx.snapshot()
            .data_as_u8_slice()
            .chunks_exact(4)
            .all(|pixel| pixel == color),
        "the image should not be affected by clearing the evicted glyphs"
    );
}

/// Rendering other scenes does not age the glyphs of a scene that has not been rendered yet, and a
/// render that is submitted before a maintenance pass is not affected by the pass, even if the pass
/// evicts the glyphs the render draws.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn glyphs_atlas_explicit_maintenance_wgpu() {
    use vello_gpu::{
        ClearSettings, RenderSize, RenderTargetConfig, Resources, Scene, TargetInit,
        TextureBindings,
    };

    const WIDTH: u16 = 300;
    const HEIGHT: u16 = 70;

    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to find an appropriate adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("Failed to create device");
    let size = wgpu::Extent3d {
        width: WIDTH.into(),
        height: HEIGHT.into(),
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let target_config = RenderTargetConfig {
        format: texture.format(),
        width: WIDTH,
        height: HEIGHT,
    };

    let render = |renderer: &mut vello_gpu::Renderer, resources: &mut Resources, scene: &Scene| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        renderer
            .render(
                scene,
                resources,
                &device,
                &queue,
                &mut encoder,
                &RenderSize {
                    width: WIDTH,
                    height: HEIGHT,
                },
                &view,
                None,
                &TextureBindings::new(),
                TargetInit::Clear(ClearSettings::default()),
            )
            .unwrap();
        encoder.finish()
    };
    let read_back = || {
        let bytes_per_row =
            (u32::from(WIDTH) * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(bytes_per_row) * u64::from(HEIGHT),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: None,
                },
            },
            size,
        );
        queue.submit([encoder.finish()]);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        buffer.slice(..).get_mapped_range().unwrap().to_vec()
    };

    let draw_glyphs = |resources: &mut Resources| {
        let mut scene = Scene::new(WIDTH, HEIGHT);
        let font_size = 50.0;
        let (font, glyphs) = layout_glyphs_roboto("Hello, world!", font_size);
        scene.set_transform(Affine::translate((0.0, f64::from(font_size))));
        scene.set_paint(BLACK);
        scene
            .glyph_run(resources, &font)
            .font_size(font_size)
            .atlas_cache(true)
            .hint(true)
            .fill_glyphs(glyphs.into_iter())
            .unwrap();
        scene
    };

    let (mut reference_renderer, mut reference_resources) =
        vello_gpu::Renderer::new(&device, &target_config);
    let reference_scene = draw_glyphs(&mut reference_resources);
    queue.submit([render(
        &mut reference_renderer,
        &mut reference_resources,
        &reference_scene,
    )]);
    let expected = read_back();
    assert!(
        expected.iter().any(|&byte| byte != 0),
        "glyphs should have been rendered"
    );

    let (mut renderer, mut resources) = vello_gpu::Renderer::new(&device, &target_config);
    resources.set_glyph_maintenance(GlyphMaintenance::Explicit);
    let scene = draw_glyphs(&mut resources);

    // The scene is rendered for the first time after many renders of another one.
    let empty = Scene::new(WIDTH, HEIGHT);
    for _ in 0..2 * glyph_eviction_passes() {
        queue.submit([render(&mut renderer, &mut resources, &empty)]);
    }
    queue.submit([render(&mut renderer, &mut resources, &scene)]);
    assert!(
        read_back() == expected,
        "rendering other scenes should not age the glyph caches"
    );

    // The glyphs are evicted by the first eviction pass at which they are older than the maximum
    // entry age. Render the scene in the frame that ends with that pass.
    let config = glifo::GlyphCacheConfig::default();
    let evicting_pass =
        (config.max_entry_age / config.eviction_frequency + 1) * config.eviction_frequency;
    for _ in 1..evicting_pass {
        queue.submit([render(&mut renderer, &mut resources, &empty)]);
        renderer.maintain_glyphs(&mut resources, &device, &queue);
    }
    queue.submit([render(&mut renderer, &mut resources, &scene)]);
    renderer.maintain_glyphs(&mut resources, &device, &queue);
    assert!(
        read_back() == expected,
        "a render submitted before a pass should not be affected by it"
    );
}
