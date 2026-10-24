// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for image upload and lookup errors in the wgpu renderer.

use std::sync::MutexGuard;

use vello_common::color::AlphaColor;
use vello_common::kurbo::Rect;
use vello_common::paint::{Image, ImageId, ImageSource};
use vello_common::peniko::ImageSampler;
use vello_common::pixmap::Pixmap;
use vello_gpu::{
    AtlasConfig, AtlasError, ClearSettings, MemorySettings, RenderError, RenderSettings,
    RenderSize, RenderTargetConfig, Renderer, Resources, Scene, TargetInit, TextureBindings,
};
use vello_tests::renderer::lock_wgpu_device;

const SIZE: u16 = 8;

struct Ctx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    renderer: Renderer,
    resources: Resources,
    _guard: MutexGuard<'static, ()>,
}

impl Ctx {
    fn new(atlas_size: u16, max_atlases: usize) -> Self {
        let (device, queue, _guard) = lock_wgpu_device();
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
                format: wgpu::TextureFormat::Rgba8Unorm,
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
            _guard,
        }
    }

    fn upload(&mut self, width: u16, height: u16) -> Result<ImageId, AtlasError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let result = self.renderer.upload_image(
            &mut self.resources,
            &self.device,
            &self.queue,
            &mut encoder,
            &Pixmap::new(width, height),
        );
        self.queue.submit([encoder.finish()]);
        result
    }
}

#[test]
fn wgpu_upload_image_larger_than_atlas_fails() {
    let mut ctx = Ctx::new(16, 1);

    assert!(matches!(
        ctx.upload(17, 4),
        Err(AtlasError::TextureTooLarge {
            width: 17,
            height: 4,
            ..
        })
    ));
}

#[test]
fn wgpu_upload_image_into_full_atlas_fails() {
    let mut ctx = Ctx::new(16, 1);
    ctx.upload(16, 16).unwrap();

    assert!(matches!(
        ctx.upload(8, 8),
        Err(AtlasError::AtlasLimitReached { max_atlases: 1, .. })
    ));
}

#[test]
fn wgpu_upload_empty_image_fails() {
    let mut ctx = Ctx::new(16, 1);

    assert!(matches!(
        ctx.upload(0, 4),
        Err(AtlasError::EmptyTexture {
            width: 0,
            height: 4
        })
    ));
}

#[test]
fn wgpu_render_with_missing_image_fails() {
    let mut ctx = Ctx::new(16, 1);
    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Image Errors Test Target"),
        size: wgpu::Extent3d {
            width: SIZE.into(),
            height: SIZE.into(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let missing = ImageId::new(7);
    let mut scene = Scene::new(SIZE, SIZE);
    scene.set_paint(Image {
        image: ImageSource::opaque_id(missing),
        sampler: ImageSampler::default(),
    });
    scene.fill_rect(&Rect::new(0.0, 0.0, f64::from(SIZE), f64::from(SIZE)));

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    let result = ctx.renderer.render(
        &scene,
        &mut ctx.resources,
        &ctx.device,
        &ctx.queue,
        &mut encoder,
        &RenderSize {
            width: SIZE,
            height: SIZE,
        },
        &view,
        None,
        &TextureBindings::new(),
        TargetInit::Clear(ClearSettings::Viewport {
            color: AlphaColor::TRANSPARENT,
        }),
    );

    assert!(matches!(result, Err(RenderError::MissingImage(id)) if id == missing));
}
