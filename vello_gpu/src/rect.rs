// Copyright 2025 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Helpers for decomposing rectangles used by the rectangle fast path.

use vello_common::geometry::RectU16;
use vello_common::kurbo::Rect;

/// The threshold of the rectangle size after which a rectangle should be split up
/// into multiple smaller ones.
const LARGE_RECT_SPLIT_THRESHOLD: u16 = 32;

/// Integer rectangle geometry and the alphas of its boundary pixels.
///
/// The alphas are the ones the strip renderer in `vello_common::rect` produces for the same
/// pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RectPart {
    /// Pixel-aligned bounds of this rectangle part.
    pub(crate) rect: RectU16,
    /// Alphas of the first column, last column, first row and last row.
    pub(crate) alphas: [u8; 4],
    /// For the top-left, top-right, bottom-left and bottom-right pixel, its alpha minus
    /// [`combine_alphas`] of its column and row alpha, which is always -1, 0 or 1.
    pub(crate) corner_corrections: [i8; 4],
}

impl RectPart {
    /// A part whose pixels are all fully covered.
    pub(crate) fn full(rect: RectU16) -> Self {
        Self {
            rect,
            alphas: [255; 4],
            corner_corrections: [0; 4],
        }
    }

    /// Whether all pixels of this part are fully covered.
    ///
    /// Edges less than half an alpha step away from the pixel grid have alphas of 255, but their
    /// corners can still round to 254.
    pub(crate) fn is_full(&self) -> bool {
        self.alphas == [255; 4] && self.corner_corrections == [0; 4]
    }

    /// The corner corrections as 2-bit two's complement values, the top-left one in the lowest
    /// bits.
    pub(crate) fn packed_corner_corrections(&self) -> u32 {
        self.corner_corrections
            .iter()
            .rev()
            .fold(0, |bits, correction| {
                (bits << 2) | u32::from(correction.cast_unsigned() & 0b11)
            })
    }

    pub(crate) fn shift(self, shift: (i32, i32)) -> Self {
        Self {
            rect: self.rect.shift(shift),
            ..self
        }
    }
}

/// The first and last pixel of a rectangle along one axis, with their coverage and alpha.
#[derive(Debug, Clone, Copy)]
struct AxisEdges {
    pixels: [u16; 2],
    coverages: [f32; 2],
    alphas: [u8; 2],
}

impl AxisEdges {
    /// The edge pixels of the span `[lo, hi]`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "recorded rect coordinates are clipped to the u16 viewport domain"
    )]
    fn new(lo: f32, hi: f32) -> Self {
        debug_assert!(0.0 <= lo && lo < hi, "invalid span [{lo}, {hi}]");
        // The span is non-negative, so truncating it floors it.
        let first = lo as u16;
        let end = hi as u16;
        let last = if f32::from(end) < hi { end } else { end - 1 };
        let coverages = [pixel_coverage(first, lo, hi), pixel_coverage(last, lo, hi)];

        Self {
            pixels: [first, last],
            coverages,
            alphas: coverages.map(coverage_to_alpha),
        }
    }
}

/// The edge pixels of a rectangle, from which its parts take their alphas.
#[derive(Debug, Clone, Copy)]
struct RectEdges {
    columns: AxisEdges,
    rows: AxisEdges,
    /// The corner corrections of the rectangle's corners, by row and column.
    corner_corrections: [[i8; 2]; 2],
}

impl RectEdges {
    fn new(columns: AxisEdges, rows: AxisEdges) -> Self {
        let mut corner_corrections = [[0; 2]; 2];
        for (row, corrections) in corner_corrections.iter_mut().enumerate() {
            for (col, correction) in corrections.iter_mut().enumerate() {
                // A corner whose column or row is fully covered has the alpha of the other one,
                // so it needs no correction.
                if columns.coverages[col] < 1.0 && rows.coverages[row] < 1.0 {
                    let alpha = coverage_to_alpha(columns.coverages[col] * rows.coverages[row]);
                    let combined = combine_alphas(columns.alphas[col], rows.alphas[row]);
                    debug_assert!(
                        alpha.abs_diff(combined) <= 1,
                        "corner alpha {alpha} is more than one away from {combined}"
                    );
                    *correction = alpha.cmp(&combined) as i8;
                }
            }
        }

        Self {
            columns,
            rows,
            corner_corrections,
        }
    }

    /// The whole rectangle as one part.
    fn whole(&self, rect: RectU16) -> RectPart {
        let [[top_left, top_right], [bottom_left, bottom_right]] = self.corner_corrections;
        RectPart {
            rect,
            alphas: [
                self.columns.alphas[0],
                self.columns.alphas[1],
                self.rows.alphas[0],
                self.rows.alphas[1],
            ],
            corner_corrections: [top_left, top_right, bottom_left, bottom_right],
        }
    }

    /// The first (0) or last (1) row as a part spanning all columns.
    ///
    /// The part is one pixel high, so the shader reads its first-row alpha and corrections; both
    /// rows of the part carry this row's.
    fn row(&self, rect: RectU16, row: usize) -> RectPart {
        let [left, right] = self.corner_corrections[row];
        RectPart {
            rect,
            alphas: [
                self.columns.alphas[0],
                self.columns.alphas[1],
                self.rows.alphas[row],
                self.rows.alphas[row],
            ],
            corner_corrections: [left, right, left, right],
        }
    }

    /// The first (0) or last (1) column as a part between the first and last row.
    fn column(&self, rect: RectU16, col: usize) -> RectPart {
        let alpha = self.columns.alphas[col];
        RectPart {
            rect,
            alphas: [alpha, alpha, 255, 255],
            corner_corrections: [0; 4],
        }
    }
}

/// The fraction of the pixel column or row `px` covered by `[lo, hi]`, as in `vello_common::rect`.
fn pixel_coverage(px: u16, lo: f32, hi: f32) -> f32 {
    let px = f32::from(px);
    (hi.min(px + 1.0) - lo.max(px)).clamp(0.0, 1.0)
}

/// Quantize a pixel's coverage to an alpha, as in `vello_common::rect`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "coverage is between 0 and 1"
)]
fn coverage_to_alpha(coverage: f32) -> u8 {
    (coverage * 255.0 + 0.5) as u8
}

/// `round(x * y / 255)`, computed like the shader does.
#[expect(clippy::cast_possible_truncation, reason = "the result is at most 255")]
fn combine_alphas(x: u8, y: u8) -> u8 {
    let p = u32::from(x) * u32::from(y) + 128;
    ((p + (p >> 8)) >> 8) as u8
}

/// A decomposed rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SplitRect {
    /// Main rectangle interior, or the complete rectangle when it is not split.
    pub(crate) main: RectPart,
    /// Top antialiased strip, if required.
    pub(crate) top: Option<RectPart>,
    /// Bottom antialiased strip, if required.
    pub(crate) bottom: Option<RectPart>,
    /// Left antialiased strip between the top and bottom strips, if required.
    pub(crate) left: Option<RectPart>,
    /// Right antialiased strip between the top and bottom strips, if required.
    pub(crate) right: Option<RectPart>,
}

/// Decompose `rect` into pixel-aligned parts, or return `None` if its edges coincide in `f32`.
///
/// Like `vello_common::rect`, this works on the edges converted to `f32`.
#[expect(
    clippy::cast_possible_truncation,
    reason = "edges are converted to f32 like in `vello_common::rect`, and recorded rect \
              coordinates are clipped to the u16 viewport domain"
)]
pub(crate) fn split_rect(rect: &Rect) -> Option<SplitRect> {
    let [x0, y0, x1, y1] = [rect.x0, rect.y0, rect.x1, rect.y1].map(|edge| edge as f32);
    // Distinct edges can be converted to the same `f32`.
    if !(x0 < x1 && y0 < y1) {
        return None;
    }

    // Pixel-aligned rectangles, such as cached glyphs, are fully covered.
    let pixels = [x0, y0, x1, y1].map(|edge| edge as u16);
    if pixels.map(f32::from) == [x0, y0, x1, y1] {
        let [x0, y0, x1, y1] = pixels;
        return Some(SplitRect {
            main: RectPart::full(RectU16::new(x0, y0, x1, y1)),
            top: None,
            bottom: None,
            left: None,
            right: None,
        });
    }

    let edges = RectEdges::new(AxisEdges::new(x0, x1), AxisEdges::new(y0, y1));
    let [x, last_x] = edges.columns.pixels;
    let [y, last_y] = edges.rows.pixels;
    let width = last_x - x + 1;
    let height = last_y - y + 1;

    // There's a balance to strike between reducing work in the fragment shader by splitting
    // out the inner part of the rectangle without anti-aliasing, and additional overhead
    // that arises from rendering 5 rectangles instead of just one. While the exact threshold
    // will obviously depend on the device, some experiments on a low-tier tablet showed that
    // `LARGE_RECT_SPLIT_THRESHOLD` seems to be a a reasonable value.
    if x1 - x0 < f32::from(LARGE_RECT_SPLIT_THRESHOLD)
        || y1 - y0 < f32::from(LARGE_RECT_SPLIT_THRESHOLD)
    {
        return Some(SplitRect {
            main: edges.whole(RectU16::new(x, y, x + width, y + height)),
            top: None,
            bottom: None,
            left: None,
            right: None,
        });
    }

    let has_left_aa = edges.columns.coverages[0] < 1.0;
    let has_right_aa = edges.columns.coverages[1] < 1.0;
    let has_top_aa = edges.rows.coverages[0] < 1.0;
    let has_bottom_aa = edges.rows.coverages[1] < 1.0;
    let has_top_strip = has_top_aa || has_left_aa || has_right_aa;
    let has_bottom_strip = has_bottom_aa || has_left_aa || has_right_aa;
    let left_inset = u16::from(has_left_aa);
    let right_inset = u16::from(has_right_aa);
    let top_inset = u16::from(has_top_strip);
    let bottom_inset = u16::from(has_bottom_strip);
    let inner_x = x + left_inset;
    let inner_y = y + top_inset;
    // Can't underflow because rectangles have at least `LARGE_RECT_SPLIT_THRESHOLD` in each
    // direction, which is larger than 2.
    let inner_width = width - left_inset - right_inset;
    let inner_height = height - top_inset - bottom_inset;

    Some(SplitRect {
        main: RectPart::full(RectU16::new(
            inner_x,
            inner_y,
            inner_x + inner_width,
            inner_y + inner_height,
        )),
        top: has_top_strip.then(|| edges.row(RectU16::new(x, y, x + width, y + 1), 0)),
        bottom: has_bottom_strip
            .then(|| edges.row(RectU16::new(x, y + height - 1, x + width, y + height), 1)),
        left: has_left_aa
            .then(|| edges.column(RectU16::new(x, inner_y, x + 1, inner_y + inner_height), 0)),
        right: has_right_aa.then(|| {
            edges.column(
                RectU16::new(x + width - 1, inner_y, x + width, inner_y + inner_height),
                1,
            )
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::{RectPart, SplitRect, split_rect};

    use alloc::vec;
    use alloc::vec::Vec;
    use vello_common::fearless_simd::Level;
    use vello_common::geometry::RectU16;
    use vello_common::kurbo::Rect;
    use vello_common::strip::visit_strip_fill_segments;
    use vello_common::tile::Tile;

    fn part(x: u16, y: u16, width: u16, height: u16, alphas: [u8; 4]) -> RectPart {
        RectPart {
            rect: RectU16::new(x, y, x + width, y + height),
            alphas,
            corner_corrections: [0; 4],
        }
    }

    fn full_part(x: u16, y: u16, width: u16, height: u16) -> RectPart {
        RectPart::full(RectU16::new(x, y, x + width, y + height))
    }

    #[test]
    fn splitter_keeps_small_rect_whole() {
        let rect = Rect::new(10.25, 20.5, 25.75, 35.25);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 16, 16, [191, 191, 128, 64]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_subpixel_rect_inside_one_pixel() {
        let rect = Rect::new(10.125, 20.25, 10.875, 20.75);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 1, 1, [191, 191, 128, 128]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_subpixel_rect_spanning_two_pixels_in_width() {
        let rect = Rect::new(10.75, 20.125, 11.25, 20.875);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 2, 1, [64, 64, 191, 191]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_subpixel_rect_spanning_two_pixels_in_height() {
        let rect = Rect::new(10.125, 20.75, 10.875, 21.25);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 1, 2, [191, 191, 64, 64]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_multi_pixel_width_rect_within_one_pixel_height() {
        let rect = Rect::new(10.25, 20.125, 14.75, 20.875);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 5, 1, [191, 191, 191, 191]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_multi_pixel_height_rect_within_one_pixel_width() {
        let rect = Rect::new(10.125, 20.25, 10.875, 24.75);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: part(10, 20, 1, 5, [191, 191, 191, 191]),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_splits_large_rect_into_five_parts() {
        let rect = Rect::new(10.25, 20.5, 42.75, 52.75);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: full_part(11, 21, 31, 31),
                top: Some(part(10, 20, 33, 1, [191, 191, 128, 128])),
                bottom: Some(part(10, 52, 33, 1, [191, 191, 191, 191])),
                left: Some(part(10, 21, 1, 31, [191, 191, 255, 255])),
                right: Some(part(42, 21, 1, 31, [191, 191, 255, 255])),
            }
        );
    }

    #[test]
    fn splitter_omits_unneeded_edge_parts() {
        let rect = Rect::new(10.0, 20.5, 42.0, 53.0);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: full_part(10, 21, 32, 32),
                top: Some(part(10, 20, 32, 1, [255, 255, 128, 128])),
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_handles_large_rect_with_only_vertical_aa() {
        let rect = Rect::new(5.0, 2.25, 37.0, 34.75);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: full_part(5, 3, 32, 31),
                top: Some(part(5, 2, 32, 1, [255, 255, 191, 191])),
                bottom: Some(part(5, 34, 32, 1, [255, 255, 191, 191])),
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_keeps_large_aligned_rect_as_single_main_rect() {
        let rect = Rect::new(10.0, 20.0, 42.0, 60.0);
        let split = split_rect(&rect).unwrap();

        assert_eq!(
            split,
            SplitRect {
                main: full_part(10, 20, 32, 40),
                top: None,
                bottom: None,
                left: None,
                right: None,
            }
        );
    }

    #[test]
    fn splitter_skips_rect_without_pixels_in_f32() {
        // Both horizontal edges are converted to the same `f32`.
        let rect = Rect::new(4300.3001, 1.0, 4300.3002, 2.0);
        assert_eq!(split_rect(&rect), None);
    }

    #[test]
    fn splitter_keeps_corner_corrections_of_near_integer_edges() {
        // All edge alphas round to 255, but the top-left alpha rounds to 254.
        let split = split_rect(&Rect::new(10.001, 20.001, 30.0, 40.0)).unwrap();
        assert_eq!(split.main.alphas, [255; 4]);
        assert_eq!(split.main.corner_corrections, [-1, 0, 0, 0]);
        assert!(!split.main.is_full());

        let split = split_rect(&Rect::new(10.0, 20.0, 30.0, 40.0)).unwrap();
        assert!(split.main.is_full());
    }

    /// The SIMD levels this machine supports.
    fn simd_levels() -> Vec<Level> {
        #[cfg_attr(
            not(any(target_arch = "x86", target_arch = "x86_64")),
            expect(unused_mut, reason = "only x86 has more than one level")
        )]
        let mut levels = vec![Level::baseline()];
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if let Some(level) = Level::try_detect() {
            levels.extend(level.as_sse4_2().map(Level::Sse4_2));
            levels.extend(level.as_avx2().map(Level::Avx2));
            levels.extend(level.as_avx512().map(Level::Avx512));
        }
        levels
    }

    /// A tile-aligned window around `rect`.
    #[expect(clippy::cast_possible_truncation, reason = "test rects are within u16")]
    fn window(rect: &Rect) -> RectU16 {
        let tile_x = |x: f64| x as u16 / Tile::WIDTH * Tile::WIDTH;
        let tile_y = |y: f64| y as u16 / Tile::HEIGHT * Tile::HEIGHT;
        RectU16::new(
            tile_x(rect.x0),
            tile_y(rect.y0),
            tile_x(rect.x1) + Tile::WIDTH,
            tile_y(rect.y1) + Tile::HEIGHT,
        )
    }

    fn index(window: RectU16, x: u16, y: u16) -> usize {
        assert!(x >= window.x0 && x < window.x1 && y >= window.y0 && y < window.y1);
        usize::from(y - window.y0) * usize::from(window.width()) + usize::from(x - window.x0)
    }

    /// The alphas the strip renderer of `vello_common::rect` produces for `rect` in `window`.
    fn strip_alphas(level: Level, rect: Rect, window: RectU16) -> Vec<u8> {
        let mut strips = Vec::new();
        let mut alphas = Vec::new();
        vello_common::rect::render(level, rect, &mut strips, &mut alphas);

        let mut image = vec![0; usize::from(window.width()) * usize::from(window.height())];
        let tile_bounds = RectU16::new(
            window.x0 / Tile::WIDTH,
            window.y0 / Tile::HEIGHT,
            window.x1 / Tile::WIDTH,
            window.y1 / Tile::HEIGHT,
        );
        visit_strip_fill_segments(
            &strips,
            tile_bounds,
            &mut image,
            |image, segment| {
                for (column, x) in (segment.x0()..segment.x1()).enumerate() {
                    for row in 0..Tile::HEIGHT {
                        let alpha_idx = segment.alpha_idx as usize
                            + column * usize::from(Tile::HEIGHT)
                            + usize::from(row);
                        image[index(window, x, segment.y() + row)] = alphas[alpha_idx];
                    }
                }
            },
            |image, segment| {
                for x in segment.x0()..segment.x1() {
                    for row in 0..Tile::HEIGHT {
                        image[index(window, x, segment.y() + row)] = 255;
                    }
                }
            },
        );
        image
    }

    /// The alphas the shader produces for the parts of `split` in `window`.
    fn quad_alphas(split: Option<SplitRect>, window: RectU16) -> Vec<u8> {
        let mut image = vec![0; usize::from(window.width()) * usize::from(window.height())];
        let mut drawn = vec![false; image.len()];
        let parts = split
            .into_iter()
            .flat_map(|split| {
                [
                    Some(split.main),
                    split.top,
                    split.bottom,
                    split.left,
                    split.right,
                ]
            })
            .flatten();
        for part in parts {
            let alphas = u32::from_le_bytes(part.alphas);
            let corner_corrections = part.packed_corner_corrections();
            let size = (part.rect.width(), part.rect.height());
            for y in part.rect.y0..part.rect.y1 {
                for x in part.rect.x0..part.rect.x1 {
                    let i = index(window, x, y);
                    assert!(!drawn[i], "pixel ({x}, {y}) is drawn twice");
                    drawn[i] = true;
                    let pixel = (x - part.rect.x0, y - part.rect.y0);
                    image[i] = rect_alpha(size, pixel, alphas, corner_corrections);
                }
            }
        }
        image
    }

    /// `rect_alpha` of `alpha.wesl`.
    fn rect_alpha(size: (u16, u16), pixel: (u16, u16), alphas: u32, corner_corrections: u32) -> u8 {
        let first = (pixel.0 == 0, pixel.1 == 0);
        let last = (pixel.0 == size.0 - 1, pixel.1 == size.1 - 1);
        let byte = |shift: u32| (alphas >> shift) & 0xff;
        let col_alpha = if first.0 {
            byte(0)
        } else if last.0 {
            byte(8)
        } else {
            255
        };
        let row_alpha = if first.1 {
            byte(16)
        } else if last.1 {
            byte(24)
        } else {
            255
        };
        let p = col_alpha * row_alpha + 128;
        let mut alpha = (p + (p >> 8)) >> 8;
        if (first.0 || last.0) && (first.1 || last.1) {
            let corner = u32::from(!first.0) + 2 * u32::from(!first.1);
            match (corner_corrections >> (2 * corner)) & 0b11 {
                0b01 => alpha += 1,
                0b11 => alpha -= 1,
                _ => {}
            }
        }
        u8::try_from(alpha).unwrap()
    }

    /// Rects with edges on a 1/16 grid or less than half an alpha step from the pixel grid, from
    /// slivers to sizes around the split threshold, near the origin and at large coordinates on
    /// one or both axes, and rects whose coverage times 255 is within a few `f32` steps of
    /// k + 0.5.
    fn rects() -> Vec<Rect> {
        let mut rects = Vec::new();
        let offsets = (0..16)
            .map(|i| f64::from(i) / 16.0)
            .chain([0.001, 0.0019, 0.001955, 0.998045, 0.999, 0.9981]);
        let sizes = [
            (0.3, 0.7),
            (0.75, 9.5),
            (1.0, 1.0),
            (1.4, 2.6),
            (9.5, 31.9),
            (31.9, 32.0),
            (32.0, 40.7),
            (32.1, 32.1),
            (40.7, 0.4),
        ];
        for (x, y) in [
            (3.0, 5.0),
            (4093.0, 5.0),
            (60001.0, 5.0),
            (60001.0, 60001.0),
        ] {
            for dx in offsets.clone() {
                for dy in offsets.clone() {
                    for (width, height) in sizes {
                        let (x0, y0) = (x + dx, y + dy);
                        rects.push(Rect::new(x0, y0, x0 + width, y0 + height));
                    }
                }
            }
        }

        // The first column, the last column, and a corner through a half-covered row, then the
        // same for rows.
        #[expect(
            clippy::cast_possible_truncation,
            reason = "rounding to f32 on purpose"
        )]
        let nudge = |edge: f64, steps: i32| {
            f64::from(f32::from_bits(
                (edge as f32).to_bits().wrapping_add_signed(steps),
            ))
        };
        for k in 0..=255 {
            let tie = (f64::from(k) + 0.5) / 255.0;
            for steps in -2..=2 {
                rects.push(Rect::new(nudge(2.0 - tie, steps), 3.0, 9.0, 12.0));
                rects.push(Rect::new(1.0, 3.0, nudge(7.0 + tie, steps), 12.0));
                rects.push(Rect::new(3.0, nudge(2.0 - tie, steps), 12.0, 9.0));
                rects.push(Rect::new(3.0, 1.0, 12.0, nudge(7.0 + tie, steps)));
                if tie < 0.5 {
                    rects.push(Rect::new(nudge(2.0 - 2.0 * tie, steps), 3.5, 9.0, 12.0));
                    rects.push(Rect::new(3.5, nudge(2.0 - 2.0 * tie, steps), 12.0, 9.0));
                }
            }
        }

        // Near edges less than half an alpha step past the pixel grid that `f32` moves further
        // in, so the corner alpha is 253.
        rects.push(Rect::new(60000.001955, 60000.001955, 60010.0, 60010.0));
        rects.push(Rect::new(4300.3001, 1.0, 4300.3002, 2.0));
        rects
    }

    #[test]
    fn split_parts_have_the_alphas_of_the_strip_renderer() {
        let rects = rects();
        for level in simd_levels() {
            for rect in &rects {
                let window = window(rect);
                let expected = strip_alphas(level, *rect, window);
                let actual = quad_alphas(split_rect(rect), window);
                if let Some(i) = expected.iter().zip(&actual).position(|(e, a)| e != a) {
                    let width = usize::from(window.width());
                    panic!(
                        "{rect:?} at {level:?}: pixel ({}, {}) has alpha {}, the strip renderer {}",
                        usize::from(window.x0) + i % width,
                        usize::from(window.y0) + i / width,
                        actual[i],
                        expected[i],
                    );
                }
            }
        }
    }
}
