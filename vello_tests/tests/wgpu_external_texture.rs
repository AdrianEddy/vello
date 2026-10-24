// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Snapshot tests for image paints backed by a native `wgpu::ExternalTexture`.
//!
//! These bypass the CPU/GPU snapshot harness because only wgpu can bind external textures, so
//! they only produce a GPU reference image. Each test returns early when the adapter does not
//! support `wgpu::Features::EXTERNAL_TEXTURE`.

use vello_common::color::palette::css::BLACK;
use vello_common::kurbo::{Affine, Rect};
use vello_common::paint::{Image, ImageSource, TextureRegion};
use vello_common::peniko::{Extend, ImageQuality, ImageSampler};
use vello_cpu::{Level, RenderMode};

use crate::renderer::{GpuRenderer, Renderer};
use crate::util::{IDENTITY_MATRIX, check_ref, create_nv12_external_texture};

/// Magnification of the 4×2 test textures, so each texel covers a visible block.
const SCALE: u16 = 16;

/// High-contrast luma checkerboard, so filtering is easy to see.
const CHECKER_Y_PLANE: [u8; 8] = [0, 255, 0, 255, 255, 0, 255, 0];
/// Left 2×2 block is blue/magenta, right block is green/yellow under [`IDENTITY_MATRIX`].
const CHECKER_UV_PLANE: [u8; 4] = [0, 255, 255, 0];

/// Luma ramp, so the out-of-range test covers both clamped and in-range values.
const RAMP_Y_PLANE: [u8; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
/// One interleaved Cb/Cr pair per 2×2 luma block.
const RAMP_UV_PLANE: [u8; 4] = [100, 110, 200, 210];

/// Each block shows its raw `[Y, Cb, Cr]` sample, so the colors show that both planes are bound
/// and each Cb/Cr pair covers a 2×2 luma block. Nearest sampling keeps the blocks hard-edged.
#[test]
fn wgpu_external_texture_nv12_nearest() {
    check_nv12(
        "wgpu_external_texture_nv12_nearest",
        &CHECKER_Y_PLANE,
        &CHECKER_UV_PLANE,
        IDENTITY_MATRIX,
        ImageQuality::Low,
        1.0,
    );
}

/// Like [`wgpu_external_texture_nv12_nearest`], but bilinear sampling blends neighboring luma
/// and chroma samples into smooth gradients.
#[test]
fn wgpu_external_texture_nv12_bilinear() {
    check_nv12(
        "wgpu_external_texture_nv12_bilinear",
        &CHECKER_Y_PLANE,
        &CHECKER_UV_PLANE,
        IDENTITY_MATRIX,
        ImageQuality::Medium,
        1.0,
    );
}

/// wgpu does not clamp the converted color, so Vello clamps it to the premultiplied range.
///
/// The texture is drawn at half opacity over black, where an unclamped red channel brighter than
/// its alpha shows up as a brighter result. The render target would clamp an opaque out-of-range
/// color anyway, and negative channels are clamped before blending, so only the overflow is
/// observable.
#[test]
fn wgpu_external_texture_nv12_out_of_range_is_clamped() {
    // R = 4Y overflows for Y > 63, G = -Y underflows, B = Y stays in range.
    #[rustfmt::skip]
    let matrix = [
        4.0, -1.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    check_nv12(
        "wgpu_external_texture_nv12_out_of_range_is_clamped",
        &RAMP_Y_PLANE,
        &RAMP_UV_PLANE,
        matrix,
        ImageQuality::Low,
        0.5,
    );
}

/// Draws a 4×2 NV12 texture magnified by [`SCALE`] over black and compares the result with the
/// `name` reference snapshot.
fn check_nv12(
    name: &str,
    y_plane: &[u8; 8],
    uv_plane: &[u8; 4],
    yuv_conversion_matrix: [f32; 16],
    quality: ImageQuality,
    alpha: f32,
) {
    let (width, height) = (4 * SCALE, 2 * SCALE);
    let mut ctx = GpuRenderer::new(
        width,
        height,
        0,
        Level::baseline(),
        RenderMode::OptimizeSpeed,
    );
    if !ctx
        .device()
        .features()
        .contains(wgpu::Features::EXTERNAL_TEXTURE)
    {
        return;
    }
    let external_texture = create_nv12_external_texture(
        ctx.device(),
        ctx.queue(),
        4,
        2,
        y_plane,
        uv_plane,
        yuv_conversion_matrix,
    );
    let texture_id = ctx.register_wgpu_external_texture(external_texture, 4, 2);
    let canvas = Rect::new(0., 0., f64::from(width), f64::from(height));
    ctx.set_paint(BLACK);
    ctx.fill_rect(&canvas);
    ctx.set_paint(Image {
        image: ImageSource::external_texture(texture_id, TextureRegion::Full, false),
        sampler: ImageSampler {
            x_extend: Extend::Pad,
            y_extend: Extend::Pad,
            quality,
            alpha,
        },
    });
    ctx.set_paint_transform(Affine::scale(f64::from(SCALE)));
    ctx.fill_rect(&canvas);
    check_ref(&mut ctx, name, name, 1, 0, true, &[]);
}
