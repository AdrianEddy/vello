// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Persistent renderer resources shared across frames.

#[cfg(feature = "text")]
use crate::text::GlyphAtlasResources;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
#[cfg(feature = "text")]
use glifo::GlyphPrepCache;
#[cfg(feature = "text")]
use vello_common::GlyphMaintenance;
use vello_common::geometry::RectU16;
use vello_common::image_cache::ImageCache;
use vello_common::multi_atlas::{AtlasConfig, AtlasId};
use vello_common::paint::ImageId;

/// Persistent resources required by Vello GPU for rendering.
///
/// A set of resources must only be used with the renderer instance associated with it.
#[derive(Debug)]
pub struct Resources {
    pub(crate) image_cache: ImageCache,
    /// Images destroyed since the last render. Ordered, so that they are freed in a reproducible
    /// order.
    destroyed_images: BTreeSet<u32>,
    #[cfg(feature = "text")]
    pub(crate) glyph_prep_cache: GlyphPrepCache,
    #[cfg(feature = "text")]
    pub(crate) glyph_resources: Option<GlyphAtlasResources>,
    #[cfg(feature = "text")]
    pub(crate) glyph_maintenance: GlyphMaintenance,
    #[cfg(feature = "text")]
    pub(crate) rendered_since_glyph_maintenance: bool,
}

impl Resources {
    pub(crate) fn new(image_atlas_config: AtlasConfig) -> Self {
        Self {
            image_cache: ImageCache::new_with_config(image_atlas_config),
            destroyed_images: BTreeSet::new(),
            #[cfg(feature = "text")]
            glyph_prep_cache: GlyphPrepCache::default(),
            // Will be initialized lazily.
            #[cfg(feature = "text")]
            glyph_resources: None,
            #[cfg(feature = "text")]
            glyph_maintenance: GlyphMaintenance::default(),
            #[cfg(feature = "text")]
            rendered_since_glyph_maintenance: false,
        }
    }

    /// Destroys an image, returning `false` if there is no image with this ID or it is already
    /// destroyed.
    ///
    /// The renderer's next `render` or `render_to_image` call clears the image's atlas region and
    /// frees it. All GPU work that reads or writes the image, including copies recorded by
    /// `upload_image`, must be submitted before that call, and neither that render nor later ones
    /// may draw the image.
    pub fn destroy_image(&mut self, image_id: ImageId) -> bool {
        self.image_cache.get(image_id).is_some() && self.destroyed_images.insert(image_id.as_u32())
    }

    /// Returns the number of allocated atlas regions: uploaded and rendered images, including
    /// destroyed ones the next render hasn't freed yet, and cached glyphs.
    pub fn atlas_allocation_count(&self) -> u32 {
        self.image_cache.atlas_manager().allocated_count()
    }

    /// Frees the destroyed images and appends their atlas regions, including padding, to
    /// `regions`, which must be cleared before anything else is written to them.
    pub(crate) fn free_destroyed_images(&mut self, regions: &mut Vec<(AtlasId, RectU16)>) {
        while let Some(id) = self.destroyed_images.pop_first() {
            if let Some(image) = self.image_cache.deallocate(ImageId::new(id)) {
                let [x, y] = image.offset;
                let padding = image.padding;
                let region = RectU16::new(
                    x - padding,
                    y - padding,
                    x + image.width + padding,
                    y + image.height + padding,
                );
                regions.push((image.atlas_id, region));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn destroy_image_reports_whether_it_queued_the_image() {
        let mut resources = Resources::new(AtlasConfig::default());
        let id = resources.image_cache.allocate(4, 4, 0).unwrap();

        assert!(resources.destroy_image(id));
        assert!(!resources.destroy_image(id), "already destroyed");
        assert!(
            resources.image_cache.get(id).is_some(),
            "stays allocated until freed"
        );

        let mut regions = Vec::new();
        resources.free_destroyed_images(&mut regions);
        assert!(resources.image_cache.get(id).is_none());
        assert!(!resources.destroy_image(id), "already freed");
        assert!(
            !resources.destroy_image(ImageId::new(99)),
            "never allocated"
        );
    }

    #[test]
    fn free_destroyed_images_frees_in_id_order_with_padding() {
        let mut resources = Resources::new(AtlasConfig::default());
        let ids = [
            resources.image_cache.allocate(4, 4, 0).unwrap(),
            resources.image_cache.allocate(6, 2, 1).unwrap(),
            resources.image_cache.allocate(3, 5, 0).unwrap(),
        ];
        let expected: Vec<_> = ids
            .iter()
            .map(|&id| {
                let image = resources.image_cache.get(id).unwrap();
                let [x, y] = image.offset;
                let p = image.padding;
                let region = RectU16::new(x - p, y - p, x + image.width + p, y + image.height + p);
                (image.atlas_id, region)
            })
            .collect();

        for &id in ids.iter().rev() {
            resources.destroy_image(id);
        }
        let mut regions = Vec::new();
        resources.free_destroyed_images(&mut regions);

        assert_eq!(regions, expected);
        resources.free_destroyed_images(&mut regions);
        assert_eq!(regions.len(), 3, "nothing is freed twice");
    }
}
