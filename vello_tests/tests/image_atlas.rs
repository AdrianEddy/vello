// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for the image/glyph atlas of the WebGL renderer.

use glifo::GlyphCacheConfig;
use vello_common::{
    color::palette::css::{GREEN, RED},
    color::{AlphaColor, Srgb},
    kurbo::{Affine, Rect},
    paint::{Image, ImageSource, TextureRegion},
    peniko::ImageSampler,
    pixmap::Pixmap,
};
use vello_gpu::{
    AtlasConfig, AtlasId, AtlasTextureInfo, MemorySettings, RenderError, RenderSettings,
    RenderSize, Resources, Scene, TextureId, WebGlError, WebGlRenderer, WebGlTextureBindings,
};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

use crate::util::layout_glyphs_roboto;

fn create_canvas() -> HtmlCanvasElement {
    web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("canvas")
        .unwrap()
        .dyn_into()
        .unwrap()
}

#[wasm_bindgen_test]
fn image_atlas_texture_is_created_on_first_upload() {
    let canvas = create_canvas();
    canvas.set_width(100);
    canvas.set_height(100);

    let atlas_config = AtlasConfig {
        initial_atlas_count: 0,
        atlas_size: (10, 10),
        ..AtlasConfig::default()
    };
    let settings = RenderSettings::default().with_memory_settings(MemorySettings {
        image_atlas_config: atlas_config,
        ..MemorySettings::default()
    });
    let (mut renderer, mut resources) = WebGlRenderer::new_with(&canvas, settings, true).unwrap();

    assert_eq!(
        renderer.atlas_info(),
        AtlasTextureInfo {
            width: 10,
            height: 10,
            texture_count: 0,
        },
        "renderer should start without allocated atlas textures"
    );
    assert_eq!(
        renderer.gl_context().get_error(),
        WebGl2RenderingContext::NO_ERROR,
        "renderer initialization should not produce a WebGL error"
    );

    renderer
        .upload_image(&mut resources, &Pixmap::new(2, 2))
        .unwrap();

    assert_eq!(
        renderer.atlas_info(),
        AtlasTextureInfo {
            width: 10,
            height: 10,
            texture_count: 1,
        },
        "first upload should allocate the first configured atlas texture"
    );
    assert_eq!(
        renderer.gl_context().get_error(),
        WebGl2RenderingContext::NO_ERROR,
        "first atlas upload should not produce a WebGL error"
    );
}

/// Uploading an image that is larger than the configured atlas must fail with
/// `AtlasError::TextureTooLarge`.
///
/// The renderer constructor configures the allocator in the returned resources, which runs the
/// `TextureTooLarge` check.
#[wasm_bindgen_test]
fn image_atlas_upload_larger_than_atlas_fails() {
    let canvas = create_canvas();
    canvas.set_width(100);
    canvas.set_height(100);

    let atlas_config = AtlasConfig {
        atlas_size: (10, 10),
        ..AtlasConfig::default()
    };
    let settings = RenderSettings::default().with_memory_settings(MemorySettings {
        image_atlas_config: atlas_config,
        ..MemorySettings::default()
    });

    let (mut renderer, mut resources) = WebGlRenderer::new_with(&canvas, settings, true).unwrap();

    // The image is much larger than the 10x10 atlas, so the upload must fail.
    let image = Pixmap::new(64, 64);
    assert!(
        matches!(
            renderer.upload_image(&mut resources, &image),
            Err(WebGlError::Render(RenderError::AtlasError(_)))
        ),
        "oversized image upload should return an atlas error"
    );
}

#[wasm_bindgen_test]
fn image_atlas_rejects_sampling_from_its_render_target() {
    let canvas = create_canvas();
    let atlas_config = AtlasConfig {
        initial_atlas_count: 1,
        atlas_size: (1, 1),
        ..AtlasConfig::default()
    };
    let settings = RenderSettings::default().with_memory_settings(MemorySettings {
        image_atlas_config: atlas_config,
        ..MemorySettings::default()
    });
    let (mut renderer, _) = WebGlRenderer::new_with(&canvas, settings, false).unwrap();
    let texture_id = TextureId(0);
    let mut bindings = WebGlTextureBindings::new();
    bindings.insert(
        texture_id,
        renderer.atlas_texture(AtlasId::new(0)).clone(),
        1,
        1,
    );

    let mut scene = Scene::new(1, 1);
    scene.set_paint(Image {
        image: ImageSource::external_texture(texture_id, TextureRegion::Full, false),
        sampler: ImageSampler::default(),
    });
    scene.fill_rect(&Rect::new(0.0, 0.0, 1.0, 1.0));

    assert!(
        matches!(
            renderer.render_to_atlas(&scene, 1, atlas_config, AtlasId::new(0), &bindings),
            Err(WebGlError::Render(RenderError::TextureFeedbackLoop(id))) if id == texture_id
        ),
        "sampling from the render target should fail"
    );
}

const SIZE: u16 = 16;

fn renderer_with_atlas(atlas_size: u16, max_atlases: usize) -> (WebGlRenderer, Resources) {
    let canvas = create_canvas();
    canvas.set_width(SIZE.into());
    canvas.set_height(SIZE.into());
    let settings = RenderSettings::default().with_memory_settings(MemorySettings {
        image_atlas_config: AtlasConfig {
            atlas_size: (atlas_size, atlas_size),
            max_atlases,
            ..AtlasConfig::default()
        },
        ..MemorySettings::default()
    });
    WebGlRenderer::new_with(&canvas, settings, false).unwrap()
}

fn solid_pixmap(size: u16, color: AlphaColor<Srgb>) -> Pixmap {
    let mut pixmap = Pixmap::new(size, size);
    pixmap.data_mut().fill(color.premultiply().to_rgba8());
    pixmap
}

fn try_render(
    renderer: &mut WebGlRenderer,
    resources: &mut Resources,
    scene: &Scene,
) -> Result<(), WebGlError> {
    let render_size = RenderSize {
        width: SIZE,
        height: SIZE,
    };
    renderer.render(
        scene,
        resources,
        &render_size,
        &WebGlTextureBindings::new(),
        AlphaColor::TRANSPARENT,
    )
}

fn render(renderer: &mut WebGlRenderer, resources: &mut Resources, scene: &Scene) {
    try_render(renderer, resources, scene).unwrap();
}

/// A scene that paints an external texture without a binding, so rendering it fails.
fn unbound_texture_scene() -> Scene {
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_paint(Image {
        image: ImageSource::external_texture(TextureId(0), TextureRegion::Full, false),
        sampler: ImageSampler::default(),
    });
    scene.fill_rect(&Rect::new(0.0, 0.0, 4.0, 4.0));
    scene
}

fn glyph_scene(resources: &mut Resources) -> Scene {
    let (font, glyphs) = layout_glyphs_roboto("Hello", 12.0);
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_transform(Affine::translate((0.0, 12.0)));
    scene
        .glyph_run(resources, &font)
        .font_size(12.0)
        .atlas_cache(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
    scene
}

/// Reads back the pixels of the currently bound framebuffer.
fn read_bound_framebuffer(gl: &WebGl2RenderingContext, width: u16, height: u16) -> Vec<u8> {
    let mut pixels = vec![0; usize::from(width) * usize::from(height) * 4];
    gl.read_pixels_with_opt_u8_array(
        0,
        0,
        width.into(),
        height.into(),
        WebGl2RenderingContext::RGBA,
        WebGl2RenderingContext::UNSIGNED_BYTE,
        Some(&mut pixels),
    )
    .unwrap();
    pixels
}

/// Reads back a whole atlas page. Leaves the canvas bound.
fn read_atlas(renderer: &WebGlRenderer, atlas_id: AtlasId) -> Vec<u8> {
    let gl = renderer.gl_context();
    let framebuffer = gl.create_framebuffer().unwrap();
    gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, Some(&framebuffer));
    gl.framebuffer_texture_2d(
        WebGl2RenderingContext::FRAMEBUFFER,
        WebGl2RenderingContext::COLOR_ATTACHMENT0,
        WebGl2RenderingContext::TEXTURE_2D,
        Some(renderer.atlas_texture(atlas_id)),
        0,
    );
    let info = renderer.atlas_info();
    let pixels = read_bound_framebuffer(gl, info.width, info.height);
    gl.bind_framebuffer(WebGl2RenderingContext::FRAMEBUFFER, None);
    gl.delete_framebuffer(Some(&framebuffer));
    pixels
}

/// The next render clears a destroyed image's region and frees it.
#[wasm_bindgen_test]
fn image_atlas_destroyed_image_is_cleared_by_next_render() {
    let (mut renderer, mut resources) = renderer_with_atlas(16, 1);
    let red = renderer
        .upload_image(&mut resources, &solid_pixmap(8, RED))
        .unwrap();
    assert!(
        read_atlas(&renderer, AtlasId::new(0))
            .iter()
            .any(|&byte| byte != 0),
        "premise: the image is in the atlas"
    );

    assert!(resources.destroy_image(red), "the image exists");
    assert!(!resources.destroy_image(red), "already destroyed");
    render(&mut renderer, &mut resources, &Scene::new(SIZE, SIZE));

    assert!(!resources.destroy_image(red), "freed");
    assert!(
        read_atlas(&renderer, AtlasId::new(0))
            .iter()
            .all(|&byte| byte == 0),
        "the freed region must be transparent"
    );
}

/// A render that fails still frees the images destroyed before it.
#[wasm_bindgen_test]
fn image_atlas_failed_render_frees_destroyed_images() {
    // A single page that fits one image, so the second upload needs the first's region.
    let (mut renderer, mut resources) = renderer_with_atlas(4, 1);
    let red = renderer
        .upload_image(&mut resources, &solid_pixmap(4, RED))
        .unwrap();
    resources.destroy_image(red);
    assert!(
        matches!(
            try_render(&mut renderer, &mut resources, &unbound_texture_scene()),
            Err(WebGlError::Render(RenderError::MissingTextureBinding(_)))
        ),
        "premise: the render fails"
    );

    assert!(
        renderer
            .upload_image(&mut resources, &solid_pixmap(4, GREEN))
            .is_ok(),
        "the destroyed image's region must be free"
    );
}

/// Destroying images and clearing evicted glyphs leave the scissor test disabled and the canvas
/// bound, so the canvas can be read back after every render.
#[wasm_bindgen_test]
fn image_atlas_atlas_clears_restore_gl_state() {
    let (mut renderer, mut resources) = renderer_with_atlas(256, 1);
    let gl = renderer.gl_context().clone();
    let red = renderer
        .upload_image(&mut resources, &solid_pixmap(8, RED))
        .unwrap();
    resources.destroy_image(red);
    assert!(
        !gl.is_enabled(WebGl2RenderingContext::SCISSOR_TEST),
        "destroying an image must not change GL state"
    );

    let scene = glyph_scene(&mut resources);
    render(&mut renderer, &mut resources, &scene);
    let mut green = Scene::new(SIZE, SIZE);
    green.set_paint(GREEN);
    green.fill_rect(&Rect::new(0.0, 0.0, f64::from(SIZE), f64::from(SIZE)));
    // Unused entries are evicted by the first eviction pass after they exceed the maximum age.
    let config = GlyphCacheConfig::default();
    for frame in 0..config.max_entry_age + config.eviction_frequency {
        render(&mut renderer, &mut resources, &green);
        assert!(
            !gl.is_enabled(WebGl2RenderingContext::SCISSOR_TEST),
            "frame {frame} left the scissor test enabled"
        );
        assert_eq!(
            read_bound_framebuffer(&gl, 1, 1),
            GREEN.premultiply().to_rgba8().to_u8_array(),
            "frame {frame} must leave the canvas bound"
        );
    }
    assert!(
        read_atlas(&renderer, AtlasId::new(0))
            .iter()
            .all(|&byte| byte == 0),
        "the image and the evicted glyphs must be cleared"
    );
}

/// Glyphs cached after an image was destroyed render as on a fresh renderer.
#[wasm_bindgen_test]
fn image_atlas_glyphs_after_destroy_match_fresh_renderer() {
    let (mut fresh, mut fresh_resources) = renderer_with_atlas(256, 1);
    let scene = glyph_scene(&mut fresh_resources);
    render(&mut fresh, &mut fresh_resources, &scene);
    let expected = read_bound_framebuffer(fresh.gl_context(), SIZE, SIZE);
    assert!(expected.iter().any(|&byte| byte != 0), "premise");

    let (mut renderer, mut resources) = renderer_with_atlas(256, 1);
    let red = renderer
        .upload_image(&mut resources, &solid_pixmap(64, RED))
        .unwrap();
    resources.destroy_image(red);
    render(&mut renderer, &mut resources, &Scene::new(SIZE, SIZE));
    let scene = glyph_scene(&mut resources);
    render(&mut renderer, &mut resources, &scene);

    assert_eq!(
        read_bound_framebuffer(renderer.gl_context(), SIZE, SIZE),
        expected,
        "the glyphs must render as on a fresh renderer"
    );
}
