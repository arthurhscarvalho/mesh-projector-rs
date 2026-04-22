use mesh_projector_core::constants::NORMAL_CHANNELS;
use mesh_projector_core::types::image_batch::ImageBatch;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WgpuOutputSelection {
    pub color: bool,
    pub depth: bool,
    pub normal: bool,
    pub occupancy: bool,
}

impl WgpuOutputSelection {
    pub const fn all() -> Self {
        Self {
            color: true,
            depth: true,
            normal: true,
            occupancy: true,
        }
    }

    pub const fn color_only() -> Self {
        Self {
            color: true,
            depth: false,
            normal: false,
            occupancy: false,
        }
    }

    pub const fn color_and_occupancy() -> Self {
        Self {
            color: true,
            depth: false,
            normal: false,
            occupancy: true,
        }
    }

    pub const fn is_empty(&self) -> bool {
        !self.color && !self.depth && !self.normal && !self.occupancy
    }
}

pub struct PackedWgpuImageOutput {
    pub selection: WgpuOutputSelection,
    pub num_views: usize,
    pub resolution: usize,
    pub image_channels: usize,
    pub color: Option<Vec<u8>>,
    pub depth: Option<Vec<u8>>,
    pub normal: Option<Vec<u8>>,
    pub occupancy: Option<Vec<u8>>,
}

impl PackedWgpuImageOutput {
    pub fn into_image_batch(self) -> ImageBatch {
        let mut image_batch = ImageBatch::new(
            self.num_views,
            self.resolution,
            self.resolution,
            self.image_channels,
            NORMAL_CHANNELS,
        );
        copy_packed_output_to_batch(&mut image_batch, &self);
        image_batch
    }
}

fn copy_packed_output_to_batch(image_batch: &mut ImageBatch, packed: &PackedWgpuImageOutput) {
    let pixels = packed.resolution * packed.resolution;
    for view_index in 0..packed.num_views {
        let scalar_start = view_index * pixels;
        let scalar_end = scalar_start + pixels;

        if let Some(color) = &packed.color {
            let src_start = view_index * pixels * 4;
            let src_end = src_start + pixels * 4;
            let images = image_batch
                .images
                .as_slice_mut()
                .expect("ImageBatch images are contiguous");
            let dst_start = view_index * pixels * packed.image_channels;
            let dst_end = dst_start + pixels * packed.image_channels;
            let image_view = &mut images[dst_start..dst_end];
            if packed.image_channels == 4 {
                image_view.copy_from_slice(&color[src_start..src_end]);
            } else {
                for (dst, src) in image_view
                    .chunks_exact_mut(3)
                    .zip(color[src_start..src_end].chunks_exact(4))
                {
                    dst.copy_from_slice(&src[0..3]);
                }
            }
        }

        if let Some(depth) = &packed.depth {
            let depth_maps = image_batch
                .depth_maps
                .as_slice_mut()
                .expect("ImageBatch depth maps are contiguous");
            depth_maps[scalar_start..scalar_end].copy_from_slice(&depth[scalar_start..scalar_end]);
        }

        if let Some(normal) = &packed.normal {
            let src_start = view_index * pixels * 4;
            let src_end = src_start + pixels * 4;
            let normal_maps = image_batch
                .normal_maps
                .as_slice_mut()
                .expect("ImageBatch normal maps are contiguous");
            let dst_start = view_index * pixels * NORMAL_CHANNELS;
            let dst_end = dst_start + pixels * NORMAL_CHANNELS;
            for (dst, src) in normal_maps[dst_start..dst_end]
                .chunks_exact_mut(NORMAL_CHANNELS)
                .zip(normal[src_start..src_end].chunks_exact(4))
            {
                dst.copy_from_slice(&src[0..NORMAL_CHANNELS]);
            }
        }

        if let Some(occupancy) = &packed.occupancy {
            let occupancy_maps = image_batch
                .occupancy_maps
                .as_slice_mut()
                .expect("ImageBatch occupancy maps are contiguous");
            occupancy_maps[scalar_start..scalar_end]
                .copy_from_slice(&occupancy[scalar_start..scalar_end]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_selection_helpers_cover_common_modes() {
        assert!(!WgpuOutputSelection::all().is_empty());
        assert!(!WgpuOutputSelection::color_only().is_empty());
        assert!(!WgpuOutputSelection::color_and_occupancy().is_empty());
        assert!(WgpuOutputSelection {
            color: false,
            depth: false,
            normal: false,
            occupancy: false,
        }
        .is_empty());
    }

    #[test]
    fn packed_color_output_converts_to_rgb_image_batch() {
        let packed = PackedWgpuImageOutput {
            selection: WgpuOutputSelection::color_only(),
            num_views: 1,
            resolution: 2,
            image_channels: 3,
            color: Some(vec![
                10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255,
            ]),
            depth: None,
            normal: None,
            occupancy: None,
        };

        let image_batch = packed.into_image_batch();

        assert_eq!(image_batch.images.shape(), &[1, 2, 2, 3]);
        assert_eq!(
            image_batch.images.as_slice().unwrap(),
            &[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        );
        assert!(image_batch.depth_maps.iter().all(|value| *value == 0));
        assert!(image_batch.normal_maps.iter().all(|value| *value == 0));
        assert!(image_batch.occupancy_maps.iter().all(|value| *value == 0));
    }

    #[test]
    fn packed_partial_maps_leave_unselected_image_batch_maps_zeroed() {
        let packed = PackedWgpuImageOutput {
            selection: WgpuOutputSelection::color_and_occupancy(),
            num_views: 2,
            resolution: 1,
            image_channels: 4,
            color: Some(vec![1, 2, 3, 4, 5, 6, 7, 8]),
            depth: None,
            normal: None,
            occupancy: Some(vec![128, 255]),
        };

        let image_batch = packed.into_image_batch();

        assert_eq!(
            image_batch.images.as_slice().unwrap(),
            &[1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(image_batch.occupancy_maps.as_slice().unwrap(), &[128, 255]);
        assert!(image_batch.depth_maps.iter().all(|value| *value == 0));
        assert!(image_batch.normal_maps.iter().all(|value| *value == 0));
    }
}
