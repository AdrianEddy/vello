// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for the image/glyph atlas of the WebGL renderer.

use glifo::GlyphCacheConfig;
use vello_common::{
    color::palette::css::{BLUE, GREEN, RED},
    color::{AlphaColor, Srgb},
    kurbo::{Affine, Rect, Shape},
    paint::{Image, ImageId, ImageSource, TextureRegion},
    peniko::ImageSampler,
    pixmap::Pixmap,
};
use vello_gpu::{
    AtlasConfig, AtlasId, AtlasTextureInfo, GlyphMaintenance, MemorySettings, RenderError,
    RenderSettings, RenderSize, Resources, Scene, TextureId, WebGlError, WebGlRenderer,
    WebGlTextureBindings,
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

/// An empty upload fails without adding atlases.
#[wasm_bindgen_test]
fn image_atlas_empty_upload_fails_without_adding_atlases() {
    let canvas = create_canvas();
    let settings = RenderSettings::default().with_memory_settings(MemorySettings {
        image_atlas_config: AtlasConfig {
            atlas_size: (10, 10),
            ..AtlasConfig::default()
        },
        ..MemorySettings::default()
    });
    let (mut renderer, mut resources) = WebGlRenderer::new_with(&canvas, settings, true).unwrap();

    for _ in 0..3 {
        assert!(
            matches!(
                renderer.upload_image(&mut resources, &Pixmap::new(0, 2)),
                Err(WebGlError::Render(RenderError::AtlasError(
                    vello_gpu::AtlasError::EmptyTexture { .. }
                )))
            ),
            "an empty upload must fail"
        );
    }
    renderer
        .upload_image(&mut resources, &Pixmap::new(2, 2))
        .unwrap();

    assert_eq!(
        renderer.atlas_info().texture_count,
        1,
        "empty uploads must not add atlases"
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

/// The canvas's top left 10×10 pixels, bottom row first.
fn image_area(gl: &WebGl2RenderingContext) -> Vec<Vec<u8>> {
    read_bound_framebuffer(gl, SIZE, SIZE)
        .chunks_exact(usize::from(SIZE) * 4)
        .skip(usize::from(SIZE) - 10)
        .map(|row| row[..40].to_vec())
        .collect()
}

/// Draws the 10×10 image `id` at the canvas's top left and returns that area.
fn draw_image(
    renderer: &mut WebGlRenderer,
    resources: &mut Resources,
    id: ImageId,
) -> Vec<Vec<u8>> {
    let mut draw = Scene::new(SIZE, SIZE);
    draw.set_paint(Image {
        image: ImageSource::opaque_id(id),
        sampler: ImageSampler::default(),
    });
    draw.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
    render(renderer, resources, &draw);
    image_area(renderer.gl_context())
}

/// Renders a 10×10 `scene` on a fresh renderer and returns the canvas's top left 10×10 pixels.
fn direct_render(scene: &Scene) -> Vec<Vec<u8>> {
    let (mut renderer, mut resources) = renderer_with_atlas(16, 1);
    render(&mut renderer, &mut resources, scene);
    image_area(renderer.gl_context())
}

fn canvas_is_bound(renderer: &WebGlRenderer) -> bool {
    renderer
        .gl_context()
        .get_parameter(WebGl2RenderingContext::FRAMEBUFFER_BINDING)
        .unwrap()
        .is_null()
}

/// An image rendered from a scene lands in its atlas region unflipped, writes nothing outside it,
/// and leaves the canvas bound.
#[wasm_bindgen_test]
fn image_atlas_render_to_image_places_and_confines() {
    // Two blue images, so that the image is placed next to other allocations.
    let (mut renderer, mut resources) = renderer_with_atlas(32, 1);
    let mut top = Pixmap::new(32, 6);
    top.data_mut().fill(BLUE.premultiply().to_rgba8());
    renderer.upload_image(&mut resources, &top).unwrap();
    let mut left = Pixmap::new(6, 26);
    left.data_mut().fill(BLUE.premultiply().to_rgba8());
    renderer.upload_image(&mut resources, &left).unwrap();

    // Red at the top left, and a green path at the bottom right that reaches past the bounds.
    let mut scene = Scene::new(10, 10);
    scene.set_paint(RED);
    scene.fill_rect(&Rect::new(0.0, 0.0, 4.0, 3.0));
    scene.set_paint(GREEN);
    scene.fill_path(&Rect::new(6.0, 5.0, 14.0, 10.0).to_path(0.1));
    let id = renderer
        .render_to_image(&scene, &mut resources, &WebGlTextureBindings::new())
        .unwrap();
    assert!(canvas_is_bound(&renderer), "the canvas must be bound");

    let page = read_atlas(&renderer, AtlasId::new(0));
    let count = |color: [u8; 4]| page.chunks_exact(4).filter(|&pixel| pixel == color).count();
    let pixel = |x: usize, y: usize| &page[(y * 32 + x) * 4..][..4];
    let red = RED.premultiply().to_rgba8().to_u8_array();
    let green = GREEN.premultiply().to_rgba8().to_u8_array();
    // The first red pixel in row order is the image's top left corner.
    let corner = page
        .chunks_exact(4)
        .position(|pixel| pixel == red)
        .expect("the image must be in the page");
    let (x, y) = (corner % 32, corner / 32);
    assert_eq!(pixel(x + 3, y + 2), red, "the red rectangle's bottom right");
    assert_eq!(pixel(x + 9, y + 9), green, "bottom right of the image");
    assert_eq!(pixel(x, y + 9), [0; 4], "bottom left of the image");
    assert_eq!(
        count(BLUE.premultiply().to_rgba8().to_u8_array()),
        32 * 6 + 6 * 26,
        "the neighbours must be intact"
    );
    assert_eq!(count(red), 4 * 3, "red pixels");
    assert_eq!(count(green), 4 * 5, "green pixels");
    assert_eq!(
        count([0; 4]),
        32 * 32 - 32 * 6 - 6 * 26 - 4 * 3 - 4 * 5,
        "everything else must stay transparent"
    );

    assert_eq!(
        draw_image(&mut renderer, &mut resources, id),
        direct_render(&scene),
        "the drawn image must match the scene"
    );
}

/// Images can be rendered, destroyed and rendered again into one region without `render`; the
/// reused region starts transparent, and the canvas stays bound.
#[wasm_bindgen_test]
fn image_atlas_render_to_image_reuses_destroyed_images() {
    let mut full = Scene::new(10, 10);
    full.set_paint(RED);
    full.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));
    let mut partial = Scene::new(10, 10);
    partial.set_paint(GREEN.with_alpha(0.5));
    partial.fill_rect(&Rect::new(0.0, 0.0, 5.0, 10.0));

    // One page that fits one image.
    let (mut renderer, mut resources) = renderer_with_atlas(10, 1);
    let mut previous = None;
    for scene in [&full, &partial, &full, &partial] {
        if let Some(id) = previous {
            assert!(resources.destroy_image(id), "the image exists");
        }
        previous = Some(
            renderer
                .render_to_image(scene, &mut resources, &WebGlTextureBindings::new())
                .unwrap(),
        );
        assert!(canvas_is_bound(&renderer), "the canvas must be bound");
    }

    assert_eq!(
        draw_image(&mut renderer, &mut resources, previous.unwrap()),
        direct_render(&partial),
        "the last image must match its scene"
    );
}

/// Failed renders into images release their allocations at once.
#[wasm_bindgen_test]
fn image_atlas_render_to_image_failures_release_the_region() {
    let missing = ImageId::new(99);
    let mut failing = Scene::new(10, 10);
    failing.set_paint(Image {
        image: ImageSource::opaque_id(missing),
        sampler: ImageSampler::default(),
    });
    failing.fill_rect(&Rect::new(0.0, 0.0, 10.0, 10.0));

    // One page that fits one image, so a leaked allocation would make the next one fail.
    let (mut renderer, mut resources) = renderer_with_atlas(10, 1);
    for _ in 0..4 {
        assert!(
            matches!(
                renderer.render_to_image(&failing, &mut resources, &WebGlTextureBindings::new()),
                Err(WebGlError::Render(RenderError::MissingImage(id))) if id == missing
            ),
            "the render must fail"
        );
        assert!(canvas_is_bound(&renderer), "the canvas must be bound");
    }
    assert_eq!(
        resources.atlas_allocation_count(),
        0,
        "the allocations must be released"
    );
    assert!(
        renderer
            .render_to_image(
                &Scene::new(10, 10),
                &mut resources,
                &WebGlTextureBindings::new()
            )
            .is_ok(),
        "the region must be free"
    );
}

/// Rendering into an image fails when the scene samples the atlas page it renders to, and the
/// allocation is released.
#[wasm_bindgen_test]
fn image_atlas_render_to_image_rejects_sampling_from_its_page() {
    let canvas = create_canvas();
    let atlas_config = AtlasConfig {
        initial_atlas_count: 1,
        max_atlases: 1,
        atlas_size: (1, 1),
        ..AtlasConfig::default()
    };
    let settings = RenderSettings::default().with_memory_settings(MemorySettings {
        image_atlas_config: atlas_config,
        ..MemorySettings::default()
    });
    let (mut renderer, mut resources) = WebGlRenderer::new_with(&canvas, settings, false).unwrap();
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
            renderer.render_to_image(&scene, &mut resources, &bindings),
            Err(WebGlError::Render(RenderError::TextureFeedbackLoop(id))) if id == texture_id
        ),
        "sampling from the target page must fail"
    );
    assert_eq!(
        resources.atlas_allocation_count(),
        0,
        "the allocation must be released"
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
    let config = GlyphCacheConfig::default();
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
        renderer.maintain_glyphs(&mut resources);
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
