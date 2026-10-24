// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Tests for confining a render to a region of the target.

use crate::renderer::Renderer;
use vello_common::color::palette::css::{BLUE, LIME, ORANGE};
use vello_common::color::{AlphaColor, Srgb};
use vello_common::kurbo::{Circle, Rect, Shape};
use vello_dev_macros::vello_test;
use vello_gpu::{ClearSettings, RectU16, RenderRegion, TargetInit};

/// Two disjoint rectangles with unaligned edges.
const REGION: &[RectU16] = &[RectU16::new(9, 13, 47, 51), RectU16::new(58, 54, 94, 91)];
const OVERLAPPING_REGION: &[RectU16] =
    &[RectU16::new(10, 10, 60, 60), RectU16::new(40, 40, 90, 90)];
const CLEAR_COLOR: AlphaColor<Srgb> = AlphaColor::from_rgb8(18, 52, 86);

/// Render the frame whose pixels a confined render must leave untouched.
fn render_previous_frame(ctx: &mut impl Renderer) {
    ctx.set_render_region(RenderRegion::Viewport);
    ctx.set_target_init(TargetInit::Clear(ClearSettings::default()));
    ctx.set_paint(BLUE);
    ctx.fill_rect(&Rect::new(20.0, 20.0, 80.0, 80.0));
    ctx.flush();
    ctx.render();
    ctx.reset();
}

fn draw_frame(ctx: &mut impl Renderer) {
    ctx.set_paint(LIME.with_alpha(0.6));
    ctx.fill_path(&Circle::new((50.0, 50.0), 42.5).to_path(0.1));
    ctx.set_paint(ORANGE);
    ctx.fill_rect(&Rect::new(4.5, 30.25, 90.75, 40.5));
}

#[vello_test(skip_webgl, gpu_no_depth)]
fn render_region_confines_viewport_clear(ctx: &mut impl Renderer) {
    render_previous_frame(ctx);
    ctx.set_render_region(RenderRegion::Rects(REGION));
    ctx.set_target_init(TargetInit::Clear(ClearSettings::Viewport {
        color: CLEAR_COLOR,
    }));
    draw_frame(ctx);
}

/// Overlapping rectangles still composite translucent content only once.
#[vello_test(skip_webgl, transparent)]
fn render_region_overlapping_rects(ctx: &mut impl Renderer) {
    render_previous_frame(ctx);
    ctx.set_render_region(RenderRegion::Rects(OVERLAPPING_REGION));
    ctx.set_target_init(TargetInit::SrcOver);
    ctx.set_paint(LIME.with_alpha(0.5));
    ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
}

/// Plain `#[test]`s comparing confined with unconfined renders, which don't run in the browser.
#[cfg(not(target_arch = "wasm32"))]
mod exact {
    use super::{OVERLAPPING_REGION, REGION, draw_frame, render_previous_frame};
    use crate::load_image;
    use crate::renderer::{GpuRenderer, Renderer};
    use crate::util::{
        IDENTITY_MATRIX, create_nv12_external_texture, get_ctx_with_depth_buffer,
        stops_blue_green_red_yellow,
    };
    use std::sync::Arc;
    use vello_common::color::palette::css::{AQUA, BLUE, ORANGE, REBECCA_PURPLE};
    use vello_common::color::{AlphaColor, Srgb};
    use vello_common::filter_effects::{EdgeMode, Filter, FilterPrimitive};
    use vello_common::kurbo::{Affine, Circle, Point, Rect, Shape, Stroke};
    use vello_common::paint::{Image, ImageSource, TextureRegion};
    use vello_common::peniko::{
        BlendMode, Compose, Extend, Gradient, ImageAlphaType, ImageQuality, ImageSampler,
        LinearGradientPosition, Mix,
    };
    use vello_common::pixmap::{PixelMetadata, Pixmap};
    use vello_cpu::RenderMode;
    use vello_gpu::{
        ClearSettings, RectU16, RenderRegion, RenderSize, RenderTargetConfig, Scene, TargetInit,
        TextureBindings, TextureId,
    };

    /// The two edges exposed by a diagonal scroll.
    const PAN_REGION: &[RectU16] = &[RectU16::new(0, 0, 100, 7), RectU16::new(93, 7, 100, 100)];
    /// Four vertical and four horizontal bars, which are split into 24 disjoint rectangles.
    const BARS_REGION: &[RectU16] = &[
        RectU16::new(5, 0, 11, 100),
        RectU16::new(29, 0, 35, 100),
        RectU16::new(53, 0, 59, 100),
        RectU16::new(77, 0, 83, 100),
        RectU16::new(0, 5, 100, 11),
        RectU16::new(0, 29, 100, 35),
        RectU16::new(0, 53, 100, 59),
        RectU16::new(0, 77, 100, 83),
    ];
    /// Rectangles that are empty once clamped to the target.
    const OUTSIDE_REGION: &[RectU16] =
        &[RectU16::new(100, 20, 140, 60), RectU16::new(30, 30, 30, 70)];
    const REGIONS: &[&[RectU16]] = &[
        &[RectU16::new(9, 13, 47, 51)],
        REGION,
        OVERLAPPING_REGION,
        PAN_REGION,
        BARS_REGION,
        OUTSIDE_REGION,
        &[],
    ];

    const TRANSLUCENT: AlphaColor<Srgb> = AlphaColor::from_rgba8(18, 52, 86, 129);
    const CLEAR_RECTS: &[RectU16] = &[RectU16::new(0, 0, 30, 30), RectU16::new(36, 44, 100, 70)];
    const TARGET_INITS: &[TargetInit<'static>] = &[
        TargetInit::SrcOver,
        TargetInit::Clear(ClearSettings::Rects {
            color: TRANSLUCENT,
            rects: CLEAR_RECTS,
        }),
        // A load op when unconfined and drawn when confined, so with colors that both store
        // exactly.
        TargetInit::Clear(ClearSettings::Viewport {
            color: AlphaColor::TRANSPARENT,
        }),
        TargetInit::Clear(ClearSettings::Viewport { color: AQUA }),
    ];

    struct Images {
        atlas: ImageSource,
        red: TextureId,
        green: TextureId,
        /// A native external texture, if the device supports them.
        nv12: Option<TextureId>,
    }

    fn solid_pixmap(r: u8, g: u8, b: u8, a: u8) -> Arc<Pixmap> {
        Arc::new(Pixmap::from_parts(
            vec![r, g, b, a],
            1,
            1,
            PixelMetadata::new(ImageAlphaType::AlphaPremultiplied, a != 255),
        ))
    }

    fn fill_texture_rect(ctx: &mut GpuRenderer, texture: TextureId, rect: Rect, opaque: bool) {
        ctx.set_paint(Image {
            image: ImageSource::external_texture(texture, TextureRegion::Full, !opaque),
            sampler: ImageSampler {
                x_extend: Extend::Pad,
                y_extend: Extend::Pad,
                quality: ImageQuality::Low,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&rect);
    }

    fn new_ctx(use_depth_buffer: bool) -> (GpuRenderer, Images) {
        let mut ctx = get_ctx_with_depth_buffer::<GpuRenderer>(
            100,
            100,
            true,
            0,
            "baseline",
            RenderMode::OptimizeSpeed,
            use_depth_buffer,
        );
        let nv12 = ctx
            .device()
            .features()
            .contains(wgpu::Features::EXTERNAL_TEXTURE)
            .then(|| {
                let texture = create_nv12_external_texture(
                    ctx.device(),
                    ctx.queue(),
                    4,
                    2,
                    &[0, 255, 0, 255, 255, 0, 255, 0],
                    &[0, 255, 255, 0],
                    IDENTITY_MATRIX,
                );
                ctx.register_wgpu_external_texture(texture, 4, 2)
            });
        let images = Images {
            atlas: ctx.get_image_source(load_image!("rgb_image_2x3")),
            red: ctx.register_external_texture(solid_pixmap(255, 0, 0, 255)),
            green: ctx.register_external_texture(solid_pixmap(0, 160, 0, 160)),
            nv12,
        };
        (ctx, images)
    }

    fn render(
        ctx: &mut GpuRenderer,
        images: &Images,
        draw: impl Fn(&mut GpuRenderer, &Images),
        target_init: TargetInit<'static>,
        region: RenderRegion<'static>,
    ) -> Pixmap {
        draw(ctx, images);
        ctx.set_target_init(target_init);
        ctx.set_render_region(region);
        ctx.render();
        let pixmap = ctx.snapshot();
        ctx.reset();
        pixmap
    }

    /// Assert that `confined` matches `unconfined` inside `region` and `previous` outside of it.
    ///
    /// Returns whether `unconfined` differs from `previous` outside of `region`, and inside of it if
    /// it covers any pixel, since only then can the two renders be told apart.
    fn assert_confined_pixels(
        previous: &Pixmap,
        unconfined: &Pixmap,
        confined: &Pixmap,
        region: &[RectU16],
        case: &str,
    ) -> bool {
        let width = usize::from(previous.width());
        let (mut has_inside, mut changed_inside, mut changed_outside) = (false, false, false);
        for (index, ((previous, unconfined), confined)) in previous
            .data()
            .iter()
            .zip(unconfined.data())
            .zip(confined.data())
            .enumerate()
        {
            let x = u16::try_from(index % width).unwrap();
            let y = u16::try_from(index / width).unwrap();
            let inside = region.iter().any(|rect| rect.contains(x, y));
            let expected = if inside { unconfined } else { previous };
            assert_eq!(confined, expected, "pixel ({x}, {y}) of {case}");

            has_inside |= inside;
            if unconfined != previous {
                changed_inside |= inside;
                changed_outside |= !inside;
            }
        }
        changed_outside && (changed_inside || !has_inside)
    }

    /// Assert that confined renders of the scene match an unconfined render inside the region and
    /// keep the previous frame outside of it.
    fn assert_confined_render_is_exact(draw: impl Fn(&mut GpuRenderer, &Images)) {
        // Covers the whole target, so that the unconfined render changes pixels everywhere.
        let draw = |ctx: &mut GpuRenderer, images: &Images| {
            ctx.set_paint(REBECCA_PURPLE.with_alpha(0.25));
            ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
            draw(ctx, images);
        };

        for use_depth_buffer in [true, false] {
            let (mut ctx, images) = new_ctx(use_depth_buffer);
            render_previous_frame(&mut ctx);
            let previous = ctx.snapshot();

            for &target_init in TARGET_INITS {
                render_previous_frame(&mut ctx);
                let unconfined =
                    render(&mut ctx, &images, draw, target_init, RenderRegion::Viewport);

                for &region in REGIONS {
                    render_previous_frame(&mut ctx);
                    let confined = render(
                        &mut ctx,
                        &images,
                        draw,
                        target_init,
                        RenderRegion::Rects(region),
                    );
                    let case =
                        format!("{target_init:?}, {region:?}, depth buffer: {use_depth_buffer}");
                    let distinguishable =
                        assert_confined_pixels(&previous, &unconfined, &confined, region, &case);
                    assert!(
                        distinguishable,
                        "{case}: the scene doesn't change the target"
                    );
                }
            }
        }
    }

    fn draw_shapes(ctx: &mut GpuRenderer, images: &Images) {
        draw_frame(ctx);
        // Large enough to be split into an opaque interior and anti-aliased edges.
        ctx.fill_rect(&Rect::new(11.25, 58.5, 49.75, 95.5));

        ctx.set_paint(Gradient {
            kind: LinearGradientPosition {
                start: Point::new(52.0, 0.0),
                end: Point::new(97.0, 0.0),
            }
            .into(),
            stops: stops_blue_green_red_yellow(),
            ..Default::default()
        });
        ctx.fill_rect(&Rect::new(52.5, 8.25, 97.0, 70.75));

        ctx.set_paint_transform(Affine::scale(3.3));
        ctx.set_paint(Image {
            image: images.atlas.clone(),
            sampler: ImageSampler {
                x_extend: Extend::Repeat,
                y_extend: Extend::Repeat,
                quality: ImageQuality::Medium,
                alpha: 1.0,
            },
        });
        ctx.fill_rect(&Rect::new(20.5, 66.0, 70.0, 98.25));
        ctx.set_paint_transform(Affine::IDENTITY);

        ctx.set_stroke(Stroke::new(2.5));
        ctx.set_paint(REBECCA_PURPLE.with_alpha(0.8));
        ctx.stroke_path(&Circle::new((50.0, 50.0), 30.0).to_path(0.1));
    }

    #[test]
    fn render_region_gpu_exact_shapes() {
        assert_confined_render_is_exact(draw_shapes);
    }

    /// Opaque and translucent draws alternating between texture bindings, so that every pass is
    /// split into several runs.
    #[test]
    fn render_region_gpu_exact_texture_runs() {
        assert_confined_render_is_exact(|ctx, images| {
            ctx.set_paint(AlphaColor::from_rgba8(32, 32, 32, 255));
            ctx.fill_rect(&Rect::new(4.0, 4.0, 96.0, 96.0));
            fill_texture_rect(ctx, images.red, Rect::new(8.0, 8.0, 48.0, 48.0), true);
            ctx.set_paint(AlphaColor::from_rgba8(255, 255, 255, 128));
            ctx.fill_rect(&Rect::new(24.0, 24.0, 72.0, 72.0));
            fill_texture_rect(ctx, images.green, Rect::new(28.0, 28.0, 68.0, 68.0), false);
            ctx.set_paint(Image {
                image: images.atlas.clone(),
                sampler: ImageSampler::default(),
            });
            ctx.fill_rect(&Rect::new(40.0, 10.0, 90.0, 40.0));
            fill_texture_rect(ctx, images.red, Rect::new(48.0, 48.0, 88.0, 88.0), true);
            if let Some(nv12) = images.nv12 {
                ctx.set_paint_transform(Affine::scale(6.0));
                fill_texture_rect(ctx, nv12, Rect::new(6.0, 60.0, 30.0, 72.0), true);
                ctx.set_paint_transform(Affine::IDENTITY);
            }
            fill_texture_rect(ctx, images.green, Rect::new(60.0, 4.0, 96.0, 30.0), false);
        });
    }

    #[test]
    fn render_region_gpu_exact_layers() {
        assert_confined_render_is_exact(|ctx, _| {
            ctx.push_opacity_layer(0.7);
            draw_frame(ctx);
            ctx.pop_layer();

            ctx.push_clip_layer(&Circle::new((70.0, 70.0), 20.0).to_path(0.1));
            ctx.set_paint(REBECCA_PURPLE);
            ctx.fill_rect(&Rect::new(40.0, 40.0, 100.0, 100.0));
            ctx.pop_layer();

            // This content lies outside of `REGION` and is only moved into it by the filter.
            ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::Offset {
                dx: -40.0,
                dy: 0.0,
            }));
            ctx.set_paint(ORANGE);
            ctx.fill_rect(&Rect::new(60.0, 16.0, 90.0, 48.0));
            ctx.pop_layer();

            ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                std_deviation: 3.0,
                edge_mode: EdgeMode::None,
            }));
            ctx.set_paint(BLUE);
            ctx.fill_rect(&Rect::new(4.0, 70.0, 56.0, 96.0));
            ctx.pop_layer();
        });
    }

    #[test]
    fn render_region_gpu_exact_translated_layers() {
        assert_confined_render_is_exact(|ctx, _| {
            ctx.set_transform(Affine::translate((17.25, -6.5)));
            ctx.push_opacity_layer(0.8);
            draw_frame(ctx);
            ctx.pop_layer();
            ctx.push_filter_layer(Filter::from_primitive(FilterPrimitive::GaussianBlur {
                std_deviation: 2.0,
                edge_mode: EdgeMode::None,
            }));
            ctx.set_paint(REBECCA_PURPLE);
            ctx.fill_rect(&Rect::new(30.0, 40.0, 70.0, 80.0));
            ctx.pop_layer();
            ctx.set_transform(Affine::IDENTITY);
        });
    }

    #[test]
    fn render_region_gpu_exact_root_blend() {
        assert_confined_render_is_exact(|ctx, _| {
            draw_frame(ctx);
            ctx.push_blend_layer(BlendMode::new(Mix::Multiply, Compose::SrcOver));
            ctx.set_paint(REBECCA_PURPLE);
            ctx.fill_rect(&Rect::new(30.0, 30.0, 80.0, 80.0));
            ctx.pop_layer();
        });
    }

    #[test]
    fn render_region_gpu_exact_root_clip_rect() {
        assert_confined_render_is_exact(|ctx, images| {
            ctx.push_clip_rect(&Rect::new(9.0, 13.0, 94.0, 91.0));
            draw_shapes(ctx, images);
            ctx.pop_clip();
        });
    }

    /// Without any draws, only the target init is confined.
    #[test]
    fn render_region_gpu_empty_scene() {
        for use_depth_buffer in [true, false] {
            let (mut ctx, images) = new_ctx(use_depth_buffer);
            render_previous_frame(&mut ctx);
            let previous = ctx.snapshot();

            for &target_init in TARGET_INITS {
                render_previous_frame(&mut ctx);
                let unconfined = render(
                    &mut ctx,
                    &images,
                    |_, _| {},
                    target_init,
                    RenderRegion::Viewport,
                );
                for &region in REGIONS {
                    render_previous_frame(&mut ctx);
                    let confined = render(
                        &mut ctx,
                        &images,
                        |_, _| {},
                        target_init,
                        RenderRegion::Rects(region),
                    );
                    assert_confined_pixels(
                        &previous,
                        &unconfined,
                        &confined,
                        region,
                        &format!("{target_init:?}, {region:?}, depth buffer: {use_depth_buffer}"),
                    );
                }
            }
        }
    }

    /// Overlapping partial renders in a row, with opaque draws that would occlude each other if
    /// depth were carried from one render to the next.
    #[test]
    fn render_region_gpu_successive_partial_renders() {
        let first = |ctx: &mut GpuRenderer, _: &Images| {
            draw_frame(ctx);
            ctx.set_paint(REBECCA_PURPLE);
            ctx.fill_rect(&Rect::new(2.0, 2.0, 98.0, 98.0));
        };
        let second = |ctx: &mut GpuRenderer, _: &Images| {
            ctx.set_paint(ORANGE);
            ctx.fill_rect(&Rect::new(0.0, 0.0, 100.0, 100.0));
            draw_frame(ctx);
        };
        let init = TargetInit::Clear(ClearSettings::Viewport { color: AQUA });

        for use_depth_buffer in [true, false] {
            let (mut ctx, images) = new_ctx(use_depth_buffer);
            render_previous_frame(&mut ctx);
            let previous = ctx.snapshot();
            let first_full = render(&mut ctx, &images, first, init, RenderRegion::Viewport);
            let second_full = render(&mut ctx, &images, second, init, RenderRegion::Viewport);

            render_previous_frame(&mut ctx);
            render(&mut ctx, &images, first, init, RenderRegion::Rects(REGION));
            let result = render(
                &mut ctx,
                &images,
                second,
                init,
                RenderRegion::Rects(OVERLAPPING_REGION),
            );

            for (index, (((result, previous), first_full), second_full)) in result
                .data()
                .iter()
                .zip(previous.data())
                .zip(first_full.data())
                .zip(second_full.data())
                .enumerate()
            {
                let x = u16::try_from(index % 100).unwrap();
                let y = u16::try_from(index / 100).unwrap();
                let inside = |region: &[RectU16]| region.iter().any(|rect| rect.contains(x, y));
                let expected = if inside(OVERLAPPING_REGION) {
                    second_full
                } else if inside(REGION) {
                    first_full
                } else {
                    previous
                };
                assert_eq!(
                    result, expected,
                    "pixel ({x}, {y}), depth buffer: {use_depth_buffer}"
                );
            }
        }
    }

    /// A confined clear of a float target stores the color of a rect clear inside the region and
    /// leaves every bit outside of it untouched.
    #[test]
    fn render_region_gpu_float_target_clear() {
        const SIZE: u16 = 100;
        const BYTES_PER_PIXEL: usize = 8;
        let format = wgpu::TextureFormat::Rgba16Float;

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
        let (mut renderer, mut resources) = vello_gpu::Renderer::new(
            &device,
            &RenderTargetConfig {
                format,
                width: SIZE,
                height: SIZE,
            },
        );
        let scene = Scene::new(SIZE, SIZE);
        let mut render = |target_init: TargetInit<'_>, region: RenderRegion<'_>| {
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
                    target_init,
                    region,
                )
                .unwrap();
            queue.submit([encoder.finish()]);
            read_texture(&device, &queue, &texture)
        };

        let rect_clear = render(
            TargetInit::Clear(ClearSettings::Rects {
                color: TRANSLUCENT,
                rects: &[RectU16::new(0, 0, SIZE, SIZE)],
            }),
            RenderRegion::Viewport,
        );
        let previous = render(
            TargetInit::Clear(ClearSettings::Viewport { color: BLUE }),
            RenderRegion::Viewport,
        );
        let confined = render(
            TargetInit::Clear(ClearSettings::Viewport { color: TRANSLUCENT }),
            RenderRegion::Rects(REGION),
        );

        for (index, ((confined, previous), rect_clear)) in confined
            .chunks_exact(BYTES_PER_PIXEL)
            .zip(previous.chunks_exact(BYTES_PER_PIXEL))
            .zip(rect_clear.chunks_exact(BYTES_PER_PIXEL))
            .enumerate()
        {
            let x = u16::try_from(index % usize::from(SIZE)).unwrap();
            let y = u16::try_from(index / usize::from(SIZE)).unwrap();
            let inside = REGION.iter().any(|rect| rect.contains(x, y));
            let expected = if inside { rect_clear } else { previous };
            assert_eq!(confined, expected, "pixel ({x}, {y})");
        }
    }

    /// Read back a texture with tightly packed rows.
    fn read_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture,
    ) -> Vec<u8> {
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

        buffer
            .slice(..)
            .get_mapped_range()
            .unwrap()
            .chunks_exact(bytes_per_row as usize)
            .flat_map(|row| &row[..row_size as usize])
            .copied()
            .collect()
    }
}
