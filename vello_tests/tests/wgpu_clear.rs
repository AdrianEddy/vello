// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for clearing wgpu render targets.

use vello_common::color::palette::css::WHITE;
use vello_common::color::{AlphaColor, Srgb};
use vello_gpu::{
    ClearSettings, RectU16, RenderRegion, RenderSize, RenderTargetConfig, Renderer, Scene,
    TargetInit, TextureBindings,
};

const SIZE: u16 = 16;
/// Its premultiplied blue component (43.506 / 255) lies close to the middle of two 8-bit levels.
const TRANSLUCENT: AlphaColor<Srgb> = AlphaColor::from_rgba8(18, 52, 86, 129);

#[test]
fn wgpu_clear_rects_use_unquantized_color_rgba8() {
    assert_rect_clear_matches_viewport_clear(wgpu::TextureFormat::Rgba8Unorm, TRANSLUCENT, 0);
}

#[test]
fn wgpu_clear_rects_use_unquantized_color_rgba16float() {
    for color in [
        TRANSLUCENT,
        AlphaColor::new([0.123_456, 0.654_321, 0.5, 0.333_333]),
    ] {
        // The load op and the fragment output may round to neighboring half floats.
        assert_rect_clear_matches_viewport_clear(wgpu::TextureFormat::Rgba16Float, color, 1);
    }
}

/// Compare a rect clear covering the whole target with a viewport clear, channel by channel in
/// units of the format's last bit.
fn assert_rect_clear_matches_viewport_clear(
    format: wgpu::TextureFormat,
    color: AlphaColor<Srgb>,
    tolerance: u16,
) {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("Failed to find an appropriate adapter");
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("Failed to create device");

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: SIZE.into(),
            height: SIZE.into(),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let (mut renderer, mut resources) = Renderer::new(
        &device,
        &RenderTargetConfig {
            format,
            width: SIZE,
            height: SIZE,
        },
    );
    let scene = Scene::new(SIZE, SIZE);

    let mut clear = |settings: ClearSettings<'_>| {
        // Start from a different color, so that a clear writing nothing is noticed.
        for settings in [ClearSettings::Viewport { color: WHITE }, settings] {
            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            renderer
                .render(
                    &scene,
                    &mut resources,
                    &device,
                    &queue,
                    &mut encoder,
                    &RenderSize {
                        width: SIZE,
                        height: SIZE,
                    },
                    &view,
                    None,
                    &TextureBindings::new(),
                    TargetInit::Clear(settings),
                    RenderRegion::Viewport,
                )
                .unwrap();
            queue.submit([encoder.finish()]);
        }
        read_channels(&device, &queue, &texture)
    };

    let viewport = clear(ClearSettings::Viewport { color });
    let rects = clear(ClearSettings::Rects {
        color,
        rects: &[RectU16::new(0, 0, SIZE, SIZE)],
    });
    for (index, (viewport, rects)) in viewport.iter().zip(&rects).enumerate() {
        assert!(
            viewport.abs_diff(*rects) <= tolerance,
            "{format:?}, {color:?}, channel {index}: viewport clear {viewport:#x}, rect clear {rects:#x}"
        );
    }
}

/// Read back the channels of an `Rgba8Unorm` or `Rgba16Float` texture.
fn read_channels(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<u16> {
    let row_size = texture.width() * texture.format().block_copy_size(None).unwrap();
    let bytes_per_row = row_size.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(bytes_per_row) * u64::from(texture.height()),
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
        texture.size(),
    );
    queue.submit([encoder.finish()]);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, |result| result.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let data = buffer.slice(..).get_mapped_range().unwrap();
    let rows = data
        .chunks_exact(bytes_per_row as usize)
        .map(|row| &row[..row_size as usize]);
    match texture.format() {
        wgpu::TextureFormat::Rgba8Unorm => rows.flatten().map(|&c| u16::from(c)).collect(),
        wgpu::TextureFormat::Rgba16Float => rows
            .flat_map(|row| row.chunks_exact(2))
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect(),
        format => unimplemented!("{format:?}"),
    }
}
