// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The color matrix filter.

/// A color matrix filter, see [`FilterPrimitive::ColorMatrix`].
///
/// [`FilterPrimitive::ColorMatrix`]: crate::filter_effects::FilterPrimitive::ColorMatrix
#[derive(Clone, Copy, Debug)]
pub struct ColorMatrix {
    /// The 4x5 matrix in row-major order, one row per output channel (`R`, `G`, `B`, `A`).
    pub matrix: [f32; 20],
}

impl ColorMatrix {
    /// Create a new color matrix filter.
    pub fn new(matrix: [f32; 20]) -> Self {
        Self { matrix }
    }
}
