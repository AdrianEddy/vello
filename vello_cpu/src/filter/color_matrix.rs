// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Color matrix filter implementation.

use super::FilterEffect;
use crate::filter::context::ScratchBuffer;
use vello_common::filter::color_matrix::ColorMatrix;
use vello_common::peniko::color::PremulRgba8;
use vello_common::pixmap::Pixmap;

impl FilterEffect for ColorMatrix {
    fn execute_lowp(&self, pixmap: &mut Pixmap, _filter_scratch: &mut ScratchBuffer) {
        apply_color_matrix(pixmap, &self.matrix);
    }

    fn execute_highp(&self, pixmap: &mut Pixmap, _filter_scratch: &mut ScratchBuffer) {
        apply_color_matrix(pixmap, &self.matrix);
    }
}

// TODO: Use SIMD, and apply per-pixel filters like this one in fine rasterization instead of on
// an intermediate pixmap.
fn apply_color_matrix(pixmap: &mut Pixmap, matrix: &[f32; 20]) {
    let pixels = pixmap.data_mut();
    if is_premul_compatible(matrix) {
        for pixel in pixels {
            *pixel = apply_premul(*pixel, matrix);
        }
    } else {
        for pixel in pixels {
            *pixel = apply_straight(*pixel, matrix);
        }
    }
}

/// Whether [`apply_premul`] can be used for the matrix.
///
/// Applying the matrix commutes with premultiplying if the color rows neither read alpha nor add
/// an offset and alpha is kept, so the result then only differs from [`apply_straight`]'s by
/// rounding.
#[inline]
fn is_premul_compatible(matrix: &[f32; 20]) -> bool {
    let (color_rows, alpha_row) = matrix.split_at(15);
    color_rows.chunks_exact(5).all(|row| {
        row[3] == 0.0
            && row[4] == 0.0
            // Weights this large could overflow in `apply_premul`'s [0, 255] units.
            && ((row[0].abs() + row[1].abs() + row[2].abs()) * 256.0).is_finite()
    }) && alpha_row == [0.0, 0.0, 0.0, 1.0, 0.0]
}

/// Apply the matrix to the unpremultiplied color of a pixel.
#[inline]
fn apply_straight(pixel: PremulRgba8, matrix: &[f32; 20]) -> PremulRgba8 {
    let a = f32::from(pixel.a) * (1.0 / 255.0);
    // The color of a fully transparent pixel is undefined; treat it as transparent black.
    let unpremultiply = if pixel.a == 0 {
        0.0
    } else {
        1.0 / f32::from(pixel.a)
    };
    let [r, g, b] = [pixel.r, pixel.g, pixel.b].map(|c| f32::from(c) * unpremultiply);
    let row = |i: usize| {
        (matrix[i] * r + matrix[i + 1] * g + matrix[i + 2] * b + matrix[i + 3] * a + matrix[i + 4])
            .clamp(0.0, 1.0)
    };
    let alpha = row(15) * 255.0;

    PremulRgba8 {
        r: round_u8(row(0) * alpha),
        g: round_u8(row(5) * alpha),
        b: round_u8(row(10) * alpha),
        a: round_u8(alpha),
    }
}

/// Apply the color rows of a matrix accepted by [`is_premul_compatible`] to the premultiplied
/// color of a pixel.
#[inline]
fn apply_premul(pixel: PremulRgba8, matrix: &[f32; 20]) -> PremulRgba8 {
    let [r, g, b, a] = [pixel.r, pixel.g, pixel.b, pixel.a].map(f32::from);
    let row = |i: usize| {
        let value = matrix[i] * r + matrix[i + 1] * g + matrix[i + 2] * b;
        // Clamping the straight color to [0, 1] is clamping the premultiplied color to [0, a].
        round_u8(value.clamp(0.0, a))
    };

    PremulRgba8 {
        r: row(0),
        g: row(5),
        b: row(10),
        a: pixel.a,
    }
}

#[inline]
fn round_u8(value: f32) -> u8 {
    // `value` is in [0, 255], so truncating after adding 0.5 rounds to the nearest integer.
    (value + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use vello_common::filter_effects::matrices;

    /// CSS `saturate(3)`: negative weights, and results beyond both clamp limits.
    const SATURATE_3: [f32; 20] = [
        2.574, -1.43, -0.144, 0.0, 0.0, //
        -0.426, 1.57, -0.144, 0.0, 0.0, //
        -0.426, -1.43, 2.856, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    const INVERT: [f32; 20] = [
        -1.0, 0.0, 0.0, 0.0, 1.0, //
        0.0, -1.0, 0.0, 0.0, 1.0, //
        0.0, 0.0, -1.0, 0.0, 1.0, //
        0.0, 0.0, 0.0, 1.0, 0.0,
    ];
    const MIXED: [f32; 20] = [
        0.9, 0.12, -0.2, 0.05, 0.15, //
        -0.3, 0.8, 0.22, 0.14, -0.07, //
        0.25, -0.11, 0.7, -0.18, 0.35, //
        0.08, 0.04, -0.06, 0.78, 0.1,
    ];

    /// The exact premultiplied result, in `[0, 255]` units, computed from the definition.
    fn exact(pixel: PremulRgba8, matrix: &[f32; 20]) -> [f64; 4] {
        let unpremultiply = |c: u8| {
            if pixel.a == 0 {
                0.0
            } else {
                f64::from(c) / f64::from(pixel.a)
            }
        };
        let input = [
            unpremultiply(pixel.r),
            unpremultiply(pixel.g),
            unpremultiply(pixel.b),
            f64::from(pixel.a) / 255.0,
        ];
        let [r, g, b, a]: [f64; 4] = core::array::from_fn(|row| {
            let weights = &matrix[row * 5..row * 5 + 5];
            let sum: f64 = (0..4).map(|i| f64::from(weights[i]) * input[i]).sum();
            (sum + f64::from(weights[4])).clamp(0.0, 1.0)
        });

        [r * a * 255.0, g * a * 255.0, b * a * 255.0, a * 255.0]
    }

    /// Pixels with every alpha class (transparent, nearly transparent, translucent, opaque) and
    /// channels at both ends of their range, plus inputs whose exact result is a rounding tie.
    fn pixels() -> impl Iterator<Item = PremulRgba8> {
        let ties = [
            PremulRgba8 {
                r: 9,
                g: 17,
                b: 10,
                a: 30,
            },
            PremulRgba8 {
                r: 2,
                g: 0,
                b: 0,
                a: 4,
            },
        ];
        [0_u8, 1, 2, 30, 128, 254, 255]
            .into_iter()
            .flat_map(|a| {
                let levels = [0, a.min(1), a / 3, a / 2, a.saturating_sub(1), a];
                levels.into_iter().flat_map(move |r| {
                    levels.into_iter().flat_map(move |g| {
                        levels.into_iter().map(move |b| PremulRgba8 { r, g, b, a })
                    })
                })
            })
            .chain(ties)
    }

    fn assert_rounds_exact_result(actual: PremulRgba8, pixel: PremulRgba8, matrix: &[f32; 20]) {
        let actual = [actual.r, actual.g, actual.b, actual.a];
        for (actual, exact) in actual.into_iter().zip(exact(pixel, matrix)) {
            assert!(
                (f64::from(actual) - exact).abs() <= 0.5 + 1e-3,
                "{pixel:?} with {matrix:?}: {actual} is not {exact} rounded"
            );
        }
    }

    #[test]
    fn both_paths_round_the_exact_result() {
        for matrix in [
            matrices::IDENTITY,
            matrices::GRAYSCALE,
            matrices::SEPIA,
            SATURATE_3,
            matrices::ALPHA_TO_BLACK,
            INVERT,
            MIXED,
        ] {
            for pixel in pixels() {
                assert_rounds_exact_result(apply_straight(pixel, &matrix), pixel, &matrix);
                if is_premul_compatible(&matrix) {
                    assert_rounds_exact_result(apply_premul(pixel, &matrix), pixel, &matrix);
                }
            }
        }
    }

    #[test]
    fn premul_compatibility() {
        assert!(is_premul_compatible(&matrices::IDENTITY));
        assert!(is_premul_compatible(&matrices::GRAYSCALE));
        assert!(is_premul_compatible(&matrices::SEPIA));
        assert!(is_premul_compatible(&SATURATE_3));
        assert!(!is_premul_compatible(&matrices::ALPHA_TO_BLACK));
        assert!(!is_premul_compatible(&INVERT));
        assert!(!is_premul_compatible(&MIXED));

        let mut with_alpha_scale = matrices::SEPIA;
        with_alpha_scale[18] = 0.5;
        assert!(!is_premul_compatible(&with_alpha_scale));
    }

    #[test]
    fn huge_weights_are_applied_to_the_straight_color() {
        let mut matrix = matrices::IDENTITY;
        matrix[..3].copy_from_slice(&[f32::MAX / 2.0, -f32::MAX / 2.0, 1.0]);
        let mut pixmap = Pixmap::new(1, 1);
        pixmap.data_mut()[0] = PremulRgba8 {
            r: 128,
            g: 128,
            b: 128,
            a: 255,
        };

        apply_color_matrix(&mut pixmap, &matrix);

        assert_eq!(pixmap.data()[0].r, 128);
    }

    #[test]
    fn offsets_add_color_to_transparent_black() {
        let matrix = [
            0.0, 0.0, 0.0, 0.0, 0.5, //
            0.0, 0.0, 0.0, 0.0, 0.25, //
            0.0, 0.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 0.0, 0.0, 0.5,
        ];
        let mut pixmap = Pixmap::new(1, 1);

        apply_color_matrix(&mut pixmap, &matrix);

        assert_eq!(
            pixmap.data()[0],
            PremulRgba8 {
                r: 64,
                g: 32,
                b: 0,
                a: 128,
            }
        );
    }
}
