// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for the image/glyph atlas configuration of the WebGL renderer.

use crate::util::layout_glyphs_roboto;
use vello_common::{
    color::AlphaColor,
    kurbo::{Affine, Rect},
    paint::{Image, ImageSource, TextureRegion},
    peniko::ImageSampler,
    pixmap::Pixmap,
};
use vello_gpu::{
    AtlasConfig, AtlasId, AtlasTextureInfo, GlyphMaintenance, MemorySettings, RenderError,
    RenderSettings, RenderSize, Scene, TextureId, WebGlError, WebGlRenderer, WebGlTextureBindings,
};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::*;
use web_sys::{HtmlCanvasElement, WebGl2RenderingContext};

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

#[wasm_bindgen_test]
fn glyph_maintenance_leaves_framebuffer_and_scissor_as_a_render_does() {
    let canvas = create_canvas();
    canvas.set_width(300);
    canvas.set_height(70);
    let (mut renderer, mut resources) = WebGlRenderer::new(&canvas).unwrap();
    resources.set_glyph_maintenance(GlyphMaintenance::Explicit);
    let render_size = RenderSize {
        width: 300,
        height: 70,
    };
    let bindings = WebGlTextureBindings::new();

    let font_size = 50.0;
    let (font, glyphs) = layout_glyphs_roboto("Hello", font_size);
    let mut scene = Scene::new(300, 70);
    scene.set_transform(Affine::translate((0.0, f64::from(font_size))));
    scene
        .glyph_run(&mut resources, &font)
        .font_size(font_size)
        .atlas_cache(true)
        .fill_glyphs(glyphs.into_iter())
        .unwrap();
    renderer
        .render(
            &scene,
            &mut resources,
            &render_size,
            &bindings,
            AlphaColor::TRANSPARENT,
        )
        .unwrap();

    // The last of these passes evicts the glyphs and clears their atlas regions.
    let config = glifo::GlyphCacheConfig::default();
    let empty = Scene::new(300, 70);
    for _ in 0..config.max_entry_age + config.eviction_frequency {
        renderer
            .render(
                &empty,
                &mut resources,
                &render_size,
                &bindings,
                AlphaColor::TRANSPARENT,
            )
            .unwrap();
        renderer.maintain_glyphs(&mut resources).unwrap();
    }

    let gl = renderer.gl_context();
    assert!(
        !gl.is_enabled(WebGl2RenderingContext::SCISSOR_TEST),
        "the scissor test should be disabled after maintenance"
    );
    assert!(
        gl.get_parameter(WebGl2RenderingContext::FRAMEBUFFER_BINDING)
            .unwrap()
            .is_null(),
        "the canvas framebuffer should be bound after maintenance"
    );
    assert_eq!(
        gl.get_error(),
        WebGl2RenderingContext::NO_ERROR,
        "maintenance should not produce a WebGL error"
    );
}
