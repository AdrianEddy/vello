// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Utility functions shared across different tests.

use crate::renderer::Renderer;
use glifo::Glyph;
use skrifa::MetadataProvider;
use skrifa::raw::FileRef;
use smallvec::smallvec;
use std::sync::Arc;
use vello_common::color::DynamicColor;
use vello_common::color::palette::css::{BLUE, GREEN, RED, WHITE, YELLOW};
use vello_common::kurbo::{BezPath, Join, Point, Rect, Shape, Stroke, Vec2};
use vello_common::peniko::{Blob, ColorStop, ColorStops, FontData};
use vello_cpu::{Level, RenderMode};

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
static REFS_PATH: std::sync::LazyLock<PathBuf> =
    std::sync::LazyLock::new(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("snapshots"));
#[cfg(not(target_arch = "wasm32"))]
static DIFFS_PATH: std::sync::LazyLock<PathBuf> =
    std::sync::LazyLock::new(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("diffs"));

/// Helper for loading png images contained within "tests/assets/**".
#[macro_export]
macro_rules! load_image {
    ($name:expr) => {{
        #[cfg(target_arch = "wasm32")]
        {
            let bytes = include_bytes!(concat!("../tests/assets/", $name, ".png"));
            std::sync::Arc::new(
                vello_common::pixmap::Pixmap::from_png(std::io::Cursor::new(bytes)).unwrap(),
            )
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("tests/assets/{}.png", $name));
            let bytes = std::fs::read(path).unwrap();
            std::sync::Arc::new(
                vello_common::pixmap::Pixmap::from_png(std::io::Cursor::new(bytes)).unwrap(),
            )
        }
    }};
}

pub(crate) fn get_ctx<T: Renderer>(
    width: u16,
    height: u16,
    transparent: bool,
    num_threads: u16,
    level: &str,
    render_mode: RenderMode,
) -> T {
    get_ctx_with_depth_buffer(
        width,
        height,
        transparent,
        num_threads,
        level,
        render_mode,
        true,
    )
}

pub(crate) fn get_ctx_with_depth_buffer<T: Renderer>(
    width: u16,
    height: u16,
    transparent: bool,
    num_threads: u16,
    level: &str,
    render_mode: RenderMode,
    use_depth_buffer: bool,
) -> T {
    let level = match level {
        #[cfg(target_arch = "aarch64")]
        "neon" => Level::Neon(
            Level::try_detect()
                .unwrap_or(Level::baseline())
                .as_neon()
                .expect("neon should be available"),
        ),
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        "wasm_simd128" => Level::WasmSimd128(
            Level::try_detect()
                .unwrap_or(Level::baseline())
                .as_wasm_simd128()
                .expect("wasm simd128 should be available"),
        ),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "sse2" => Level::Sse2(
            Level::try_detect()
                .and_then(Level::as_sse2)
                .expect("SSE2 should be available"),
        ),
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "sse42" => {
            if std::arch::is_x86_feature_detected!("sse4.2") {
                Level::Sse4_2(unsafe { fearless_simd::Sse4_2::assume_supported() })
            } else {
                panic!("sse4.2 feature not detected");
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "avx2" => {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                Level::Avx2(unsafe { fearless_simd::Avx2::assume_supported() })
            } else {
                panic!("avx2 or fma feature not detected");
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        "avx512" => Level::Avx512(
            Level::try_detect()
                .and_then(Level::as_avx512)
                .expect("Ice Lake AVX-512 should be available"),
        ),
        #[cfg(feature = "force_support_fallback")]
        "fallback" => Level::fallback(),
        "baseline" => Level::baseline(),
        _ => panic!("unknown level: {level}"),
    };

    let mut ctx = T::new_with_depth_buffer(
        width,
        height,
        num_threads,
        level,
        render_mode,
        use_depth_buffer,
    );

    if !transparent {
        let path = Rect::new(0.0, 0.0, width as f64, height as f64).to_path(0.1);

        ctx.set_paint(WHITE);
        ctx.fill_path(&path);
    }

    ctx
}

pub(crate) fn miter_stroke_2() -> Stroke {
    Stroke {
        width: 2.0,
        join: Join::Miter,
        ..Default::default()
    }
}

pub(crate) fn crossed_line_star() -> BezPath {
    let mut path = BezPath::new();
    path.move_to((50.0, 10.0));
    path.line_to((75.0, 90.0));
    path.line_to((10.0, 40.0));
    path.line_to((90.0, 40.0));
    path.line_to((25.0, 90.0));
    path.line_to((50.0, 10.0));

    path
}

pub(crate) fn circular_star(center: Point, n: usize, inner: f64, outer: f64) -> BezPath {
    let mut path = BezPath::new();
    let start_angle = -std::f64::consts::FRAC_PI_2;
    path.move_to(center + outer * Vec2::from_angle(start_angle));
    for i in 1..n * 2 {
        let th = start_angle + i as f64 * std::f64::consts::PI / n as f64;
        let r = if i % 2 == 0 { outer } else { inner };
        path.line_to(center + r * Vec2::from_angle(th));
    }
    path.close_path();
    path
}

pub(crate) fn layout_glyphs_roboto(text: &str, font_size: f32) -> (FontData, Vec<Glyph>) {
    const ROBOTO_FONT: &[u8] = include_bytes!("../../assets/roboto/Roboto-Regular.ttf");
    let font = FontData::new(Blob::new(Arc::new(ROBOTO_FONT)), 0);

    layout_glyphs(text, font_size, font)
}

pub(crate) fn layout_glyphs_noto_cbtf(text: &str, font_size: f32) -> (FontData, Vec<Glyph>) {
    const NOTO_FONT: &[u8] =
        include_bytes!("../../assets/noto_color_emoji/NotoColorEmoji-CBTF-Subset.ttf");
    let font = FontData::new(Blob::new(Arc::new(NOTO_FONT)), 0);

    layout_glyphs(text, font_size, font)
}

pub(crate) fn layout_glyphs_noto_colr(text: &str, font_size: f32) -> (FontData, Vec<Glyph>) {
    const NOTO_FONT: &[u8] =
        include_bytes!("../../assets/noto_color_emoji/NotoColorEmoji-Subset.ttf");
    let font = FontData::new(Blob::new(Arc::new(NOTO_FONT)), 0);

    layout_glyphs(text, font_size, font)
}

#[cfg(target_os = "macos")]
pub(crate) fn layout_glyphs_apple_color_emoji(
    text: &str,
    font_size: f32,
) -> (FontData, Vec<Glyph>) {
    let apple_font: Vec<u8> = std::fs::read("/System/Library/Fonts/Apple Color Emoji.ttc").unwrap();
    let font = FontData::new(Blob::new(Arc::new(apple_font)), 0);

    layout_glyphs(text, font_size, font)
}

/// ***DO NOT USE THIS OUTSIDE OF THESE TESTS***
///
/// This function is used for _TESTING PURPOSES ONLY_. If you need to layout and shape
/// text for your application, use a proper text shaping library like `Parley`.
///
/// We use this function as a convenience for testing; to get some glyphs shaped and laid
/// out in a small amount of code without having to go through the trouble of setting up a
/// full text layout pipeline, which you absolutely should do in application code.
fn layout_glyphs(text: &str, font_size: f32, font: FontData) -> (FontData, Vec<Glyph>) {
    let font_ref = {
        let file_ref = FileRef::new(font.data.as_ref()).unwrap();
        match file_ref {
            FileRef::Font(f) => f,
            FileRef::Collection(collection) => collection.get(font.index).unwrap(),
        }
    };
    let font_size = skrifa::instance::Size::new(font_size);
    let axes = font_ref.axes();
    let variations: Vec<(&str, f32)> = vec![];
    let var_loc = axes.location(variations.as_slice());
    let charmap = font_ref.charmap();
    let metrics = font_ref.metrics(font_size, &var_loc);
    let line_height = metrics.ascent - metrics.descent + metrics.leading;
    let glyph_metrics = font_ref.glyph_metrics(font_size, &var_loc);

    let mut pen_x = 0_f32;
    let mut pen_y = 0_f32;

    let glyphs = text
        .chars()
        .filter_map(|ch| {
            if ch == '\n' {
                pen_y += line_height;
                pen_x = 0.0;
                return None;
            }
            let gid = charmap.map(ch).unwrap_or_default();
            let advance = glyph_metrics.advance_width(gid).unwrap_or_default();
            let x = pen_x;
            pen_x += advance;
            Some(Glyph {
                id: gid.to_u32(),
                x,
                y: pen_y,
            })
        })
        .collect::<Vec<_>>();

    (font, glyphs)
}

pub(crate) fn stops_green_blue() -> ColorStops {
    ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(GREEN),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(BLUE),
        },
    ])
}

pub(crate) fn stops_green_blue_with_alpha() -> ColorStops {
    ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(GREEN.with_alpha(0.25)),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(BLUE.with_alpha(0.75)),
        },
    ])
}

pub(crate) fn stops_blue_green_red_yellow() -> ColorStops {
    ColorStops(smallvec![
        ColorStop {
            offset: 0.0,
            color: DynamicColor::from_alpha_color(BLUE),
        },
        ColorStop {
            offset: 0.33,
            color: DynamicColor::from_alpha_color(GREEN),
        },
        ColorStop {
            offset: 0.66,
            color: DynamicColor::from_alpha_color(RED),
        },
        ColorStop {
            offset: 1.0,
            color: DynamicColor::from_alpha_color(YELLOW),
        },
    ])
}

pub(crate) fn check_ref(
    ctx: &mut impl Renderer,
    test_name: &str,
    specific_name: &str,
    threshold: u8,
    diff_pixels: u32,
    is_reference: bool,
    _ref_data: &[u8],
) {
    #[cfg(target_arch = "wasm32")]
    assert!(!is_reference, "WASM cannot create new reference images");

    ctx.render();
    let encoded_image = ctx.snapshot().into_png().unwrap();
    vello_test_support::Snapshot {
        reference_name: test_name,
        run_name: specific_name,
        threshold,
        diff_pixels,
        is_reference,
        #[cfg(not(target_arch = "wasm32"))]
        snapshots_dir: &REFS_PATH,
        #[cfg(not(target_arch = "wasm32"))]
        diffs_dir: &DIFFS_PATH,
        #[cfg(target_arch = "wasm32")]
        reference_png: _ref_data,
    }
    .check(&encoded_image);
}

/// Maps each pixel of an NV12 external texture to its raw `[Y, Cb, Cr]` sample.
#[cfg(not(target_arch = "wasm32"))]
#[rustfmt::skip]
pub(crate) const IDENTITY_MATRIX: [f32; 16] = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
];

/// Creates a two-plane NV12 external texture: a full-resolution `R8Unorm` Y plane and a
/// half-resolution `Rg8Unorm` interleaved Cb/Cr plane, converted with `yuv_conversion_matrix`.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn create_nv12_external_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    y_plane: &[u8],
    uv_plane: &[u8],
    yuv_conversion_matrix: [f32; 16],
) -> wgpu::ExternalTexture {
    let y_view = create_plane_view(
        device,
        queue,
        wgpu::TextureFormat::R8Unorm,
        width,
        height,
        y_plane,
        width,
    );
    let uv_view = create_plane_view(
        device,
        queue,
        wgpu::TextureFormat::Rg8Unorm,
        width / 2,
        height / 2,
        uv_plane,
        width,
    );
    #[rustfmt::skip]
    let identity_gamut = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    device.create_external_texture(
        &wgpu::ExternalTextureDescriptor {
            label: Some("Test NV12 External Texture"),
            width,
            height,
            format: wgpu::ExternalTextureFormat::Nv12,
            yuv_conversion_matrix,
            gamut_conversion_matrix: identity_gamut,
            src_transfer_function: wgpu::ExternalTextureTransferFunction::default(),
            dst_transfer_function: wgpu::ExternalTextureTransferFunction::default(),
            sample_transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            load_transform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        },
        &[&y_view, &uv_view],
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn create_plane_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    data: &[u8],
    bytes_per_row: u32,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Test External Texture Plane"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: None,
        },
        size,
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}
