// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for freeing and reusing image atlas regions in the wgpu renderer.

use std::sync::MutexGuard;

use glifo::GlyphCacheConfig;
use vello_common::color::palette::css::{BLUE, GREEN, RED};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Affine, Rect};
use vello_common::paint::{Image, ImageId, ImageSource, TextureRegion};
use vello_common::peniko::ImageSampler;
use vello_common::pixmap::Pixmap;
use vello_gpu::{
    AtlasConfig, AtlasId, ClearSettings, MemorySettings, RenderError, RenderSettings, RenderSize,
    RenderTargetConfig, Renderer, Resources, Scene, TargetInit, TextureBindings, TextureId,
};
use vello_tests::renderer::lock_wgpu_device;

use crate::util::layout_glyphs_roboto;

const SIZE: u16 = 32;
const TRANSPARENT: [u8; 4] = [0; 4];

struct Ctx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    resources: Resources,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    _guard: MutexGuard<'static, ()>,
}

impl Ctx {
    fn new(atlas_size: u16, max_atlases: usize) -> Self {
        let (device, queue, _guard) = lock_wgpu_device();
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atlas Test Target"),
            size: wgpu::Extent3d {
                width: SIZE.into(),
                height: SIZE.into(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&wgpu::TextureViewDescriptor::default());
        let settings = RenderSettings::default().with_memory_settings(MemorySettings {
            image_atlas_config: AtlasConfig {
                atlas_size: (atlas_size, atlas_size),
                max_atlases,
                ..AtlasConfig::default()
            },
            ..MemorySettings::default()
        });
        let (renderer, resources) = Renderer::new_with(
            &device,
            &RenderTargetConfig {
                format: target.format(),
                width: SIZE,
                height: SIZE,
            },
            settings,
        );
        Self {
            device,
            queue,
            renderer,
            resources,
            target,
            view,
            _guard,
        }
    }

    fn encoder(&self) -> wgpu::CommandEncoder {
        self.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default())
    }

    fn upload(&mut self, size: u16, color: AlphaColor<Srgb>) -> ImageId {
        let mut encoder = self.encoder();
        let id = self.renderer.upload_image(
            &mut self.resources,
            &self.device,
            &self.queue,
            &mut encoder,
            &solid_pixmap(size, color),
        );
        self.queue.submit([encoder.finish()]);
        id
    }

    /// Uploads through the texture writer, which records a copy into `encoder`.
    fn upload_texture(
        &mut self,
        size: u16,
        color: AlphaColor<Srgb>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> ImageId {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Atlas Test Source"),
            size: wgpu::Extent3d {
                width: size.into(),
                height: size.into(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let pixmap = solid_pixmap(size, color);
        self.queue.write_texture(
            texture.as_image_copy(),
            pixmap.data_as_u8_slice(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(u32::from(size) * 4),
                rows_per_image: None,
            },
            texture.size(),
        );
        self.renderer.upload_image(
            &mut self.resources,
            &self.device,
            &self.queue,
            encoder,
            &texture,
        )
    }

    fn try_encode_render(
        &mut self,
        scene: &Scene,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), RenderError> {
        self.renderer.render(
            scene,
            &mut self.resources,
            &self.device,
            &self.queue,
            encoder,
            &RenderSize {
                width: SIZE,
                height: SIZE,
            },
            &self.view,
            None,
            &TextureBindings::new(),
            TargetInit::Clear(ClearSettings::Viewport {
                color: AlphaColor::TRANSPARENT,
            }),
        )
    }

    fn encode_render(&mut self, scene: &Scene, encoder: &mut wgpu::CommandEncoder) {
        self.try_encode_render(scene, encoder).unwrap();
    }

    fn render(&mut self, scene: &Scene) {
        let mut encoder = self.encoder();
        self.encode_render(scene, &mut encoder);
        self.queue.submit([encoder.finish()]);
    }

    /// Reads back an `Rgba8Unorm` texture, row by row.
    fn read(&self, texture: &wgpu::Texture) -> Vec<[u8; 4]> {
        let (width, height) = (texture.width(), texture.height());
        let bytes_per_row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Atlas Test Readback"),
            size: u64::from(bytes_per_row * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.encoder();
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
            texture.size(),
        );
        self.queue.submit([encoder.finish()]);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        data.chunks_exact(bytes_per_row as usize)
            .flat_map(|row| row[..(width * 4) as usize].chunks_exact(4))
            .map(|pixel| [pixel[0], pixel[1], pixel[2], pixel[3]])
            .collect()
    }

    fn read_target(&self) -> Vec<[u8; 4]> {
        self.read(&self.target)
    }

    fn read_atlas(&self, atlas_id: u32) -> Vec<[u8; 4]> {
        self.read(self.renderer.atlas_texture(AtlasId::new(atlas_id)))
    }
}

fn solid_pixmap(size: u16, color: AlphaColor<Srgb>) -> Pixmap {
    let mut pixmap = Pixmap::new(size, size);
    pixmap.data_mut().fill(color.premultiply().to_rgba8());
    pixmap
}

fn premul(color: AlphaColor<Srgb>) -> [u8; 4] {
    color.premultiply().to_rgba8().to_u8_array()
}

/// A scene that draws `id`, of size `size`, at the origin.
fn image_scene(id: ImageId, size: u16) -> Scene {
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_paint(Image {
        image: ImageSource::opaque_id(id),
        sampler: ImageSampler::default(),
    });
    scene.fill_rect(&Rect::new(0.0, 0.0, f64::from(size), f64::from(size)));
    scene
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

/// The target after rendering `glyph_scene` on a fresh renderer.
fn fresh_glyph_render(atlas_size: u16) -> Vec<[u8; 4]> {
    let mut ctx = Ctx::new(atlas_size, 1);
    let scene = glyph_scene(&mut ctx.resources);
    ctx.render(&scene);
    let pixels = ctx.read_target();
    assert!(
        count(&pixels, TRANSPARENT) < pixels.len(),
        "premise: the glyphs are drawn"
    );
    pixels
}

fn count(pixels: &[[u8; 4]], color: [u8; 4]) -> usize {
    pixels.iter().filter(|&&pixel| pixel == color).count()
}

/// Destroying an image and caching glyphs, then rendering them on the same encoder, draws the
/// glyphs.
#[test]
fn wgpu_atlas_destroy_then_cache_glyphs_on_one_encoder() {
    let expected = fresh_glyph_render(256);

    let mut ctx = Ctx::new(256, 1);
    let red = ctx.upload(64, RED);
    let mut encoder = ctx.encoder();
    assert!(ctx.resources.destroy_image(red));
    let scene = glyph_scene(&mut ctx.resources);
    ctx.encode_render(&scene, &mut encoder);
    ctx.queue.submit([encoder.finish()]);

    assert_eq!(ctx.read_target(), expected);
}

/// An image uploaded into the region freed by a render isn't erased when that render's encoder
/// is submitted.
#[test]
fn wgpu_atlas_upload_into_freed_region() {
    // A single page that fits one image, so the second upload must reuse the first's region.
    let mut ctx = Ctx::new(4, 1);
    let red = ctx.upload(4, RED);
    ctx.resources.destroy_image(red);
    let mut encoder = ctx.encoder();
    ctx.encode_render(&Scene::new(SIZE, SIZE), &mut encoder);
    let green = ctx.upload(4, GREEN);
    ctx.queue.submit([encoder.finish()]);

    ctx.render(&image_scene(green, 4));
    assert_eq!(ctx.read_target()[0], premul(GREEN));
}

/// Like [`wgpu_atlas_upload_into_freed_region`], with a texture copy recorded into a separate
/// encoder that is submitted first.
#[test]
fn wgpu_atlas_texture_upload_into_freed_region() {
    let mut ctx = Ctx::new(4, 1);
    let red = ctx.upload(4, RED);
    ctx.resources.destroy_image(red);
    let mut encoder = ctx.encoder();
    ctx.encode_render(&Scene::new(SIZE, SIZE), &mut encoder);
    let mut upload_encoder = ctx.encoder();
    let green = ctx.upload_texture(4, GREEN, &mut upload_encoder);
    ctx.queue.submit([upload_encoder.finish()]);
    ctx.queue.submit([encoder.finish()]);

    ctx.render(&image_scene(green, 4));
    assert_eq!(ctx.read_target()[0], premul(GREEN));
}

/// A render that fails still frees the images destroyed before it.
#[test]
fn wgpu_atlas_failed_render_frees_destroyed_images() {
    let mut ctx = Ctx::new(4, 1);
    let red = ctx.upload(4, RED);
    ctx.resources.destroy_image(red);
    let mut encoder = ctx.encoder();
    let result = ctx.try_encode_render(&unbound_texture_scene(), &mut encoder);
    assert!(
        matches!(result, Err(RenderError::MissingTextureBinding(_))),
        "premise: the render fails"
    );
    ctx.queue.submit([encoder.finish()]);

    let green = ctx.upload(4, GREEN);
    ctx.render(&image_scene(green, 4));
    assert_eq!(ctx.read_target()[0], premul(GREEN));
}

/// A render encoded before `destroy_image` still draws the image when it is submitted
/// afterwards.
#[test]
fn wgpu_atlas_destroyed_image_survives_encoded_render() {
    let mut ctx = Ctx::new(64, 1);
    let red = ctx.upload(8, RED);
    let mut encoder = ctx.encoder();
    ctx.encode_render(&image_scene(red, 8), &mut encoder);
    ctx.resources.destroy_image(red);
    ctx.queue.submit([encoder.finish()]);

    assert_eq!(ctx.read_target()[0], premul(RED));
}

/// The next render clears exactly the destroyed images' regions, on every page, and frees them.
#[test]
fn wgpu_atlas_destroyed_images_are_cleared_by_next_render() {
    // Four 8×8 images fill a page, so the first six images span two pages.
    let mut ctx = Ctx::new(16, 2);
    let colors = [BLUE, RED, RED, GREEN, RED, BLUE];
    let ids = colors.map(|color| ctx.upload(8, color));
    let destroyed = [ids[1], ids[2], ids[4]];
    for id in destroyed {
        assert!(ctx.resources.destroy_image(id));
    }
    ctx.render(&Scene::new(SIZE, SIZE));

    for id in destroyed {
        assert!(!ctx.resources.destroy_image(id), "freed");
    }
    let page_0 = ctx.read_atlas(0);
    assert_eq!(count(&page_0, premul(BLUE)), 64);
    assert_eq!(count(&page_0, premul(GREEN)), 64);
    assert_eq!(count(&page_0, TRANSPARENT), 128);
    let page_1 = ctx.read_atlas(1);
    assert_eq!(count(&page_1, premul(BLUE)), 64);
    assert_eq!(count(&page_1, TRANSPARENT), 192);
}

/// Glyphs cached into a freed region render as on a fresh renderer, so the region was left
/// transparent around them.
#[test]
fn wgpu_atlas_glyphs_in_freed_region() {
    let expected = fresh_glyph_render(256);

    let mut ctx = Ctx::new(256, 1);
    let red = ctx.upload(64, RED);
    ctx.resources.destroy_image(red);
    ctx.render(&Scene::new(SIZE, SIZE));
    let scene = glyph_scene(&mut ctx.resources);
    ctx.render(&scene);

    assert_eq!(ctx.read_target(), expected);
}

/// Evicting cached glyphs clears them from the atlas.
#[test]
fn wgpu_atlas_evicted_glyphs_are_cleared() {
    let mut ctx = Ctx::new(256, 1);
    let scene = glyph_scene(&mut ctx.resources);
    ctx.render(&scene);
    assert!(
        count(&ctx.read_atlas(0), TRANSPARENT) < 256 * 256,
        "premise: the glyphs are cached"
    );

    // Unused entries are evicted by the first eviction pass after they exceed the maximum age.
    let config = GlyphCacheConfig::default();
    for _ in 0..config.max_entry_age + config.eviction_frequency {
        ctx.render(&Scene::new(SIZE, SIZE));
    }

    assert_eq!(count(&ctx.read_atlas(0), TRANSPARENT), 256 * 256);
}
