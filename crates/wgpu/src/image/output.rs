use mesh_projector_core::types::image_batch::ImageBatch;

use crate::image::resources::WgpuReadbackLeaseGuard;

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
    pub color: Option<Vec<u8>>,
    pub depth: Option<Vec<u8>>,
    pub normal: Option<Vec<u8>>,
    pub occupancy: Option<Vec<u8>>,
}

#[derive(Clone, Copy, Debug)]
pub struct PlaneView<'a> {
    pub bytes: &'a [u8],
    pub bytes_per_view: usize,
    pub stride_bytes: usize,
}

impl<'a> PlaneView<'a> {
    fn copy_views_to(&self, dst: &mut [u8], num_views: usize) {
        for view_index in 0..num_views {
            let src_start = view_index * self.stride_bytes;
            let dst_start = view_index * self.bytes_per_view;
            let src_end = src_start + self.bytes_per_view;
            let dst_end = dst_start + self.bytes_per_view;
            dst[dst_start..dst_end].copy_from_slice(&self.bytes[src_start..src_end]);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct WgpuMappedImageReadback<'a> {
    pub selection: WgpuOutputSelection,
    pub num_views: usize,
    pub resolution: usize,
    pub color: Option<PlaneView<'a>>,
    pub depth: Option<PlaneView<'a>>,
    pub normal: Option<PlaneView<'a>>,
    pub occupancy: Option<PlaneView<'a>>,
}

pub struct OwnedPlaneView {
    storage: OwnedPlaneStorage,
    pub bytes_per_view: usize,
    pub stride_bytes: usize,
}

enum OwnedPlaneStorage {
    Mapped(wgpu::BufferView),
    Bytes(Vec<u8>),
}

impl OwnedPlaneView {
    pub(crate) fn mapped(
        view: wgpu::BufferView,
        bytes_per_view: usize,
        stride_bytes: usize,
    ) -> Self {
        Self {
            storage: OwnedPlaneStorage::Mapped(view),
            bytes_per_view,
            stride_bytes,
        }
    }

    pub(crate) fn owned_bytes(bytes: Vec<u8>, bytes_per_view: usize, stride_bytes: usize) -> Self {
        Self {
            storage: OwnedPlaneStorage::Bytes(bytes),
            bytes_per_view,
            stride_bytes,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        match &self.storage {
            OwnedPlaneStorage::Mapped(view) => &view[..],
            OwnedPlaneStorage::Bytes(bytes) => bytes,
        }
    }

    pub fn as_plane_view(&self) -> PlaneView<'_> {
        PlaneView {
            bytes: self.bytes(),
            bytes_per_view: self.bytes_per_view,
            stride_bytes: self.stride_bytes,
        }
    }
}

pub struct WgpuOwnedMappedImageReadback {
    pub selection: WgpuOutputSelection,
    pub num_views: usize,
    pub resolution: usize,
    color: Option<OwnedPlaneView>,
    depth: Option<OwnedPlaneView>,
    normal: Option<OwnedPlaneView>,
    occupancy: Option<OwnedPlaneView>,
    readback_lease: Option<WgpuReadbackLeaseGuard>,
}

impl WgpuOwnedMappedImageReadback {
    pub(crate) fn new(
        selection: WgpuOutputSelection,
        num_views: usize,
        resolution: usize,
        color: Option<OwnedPlaneView>,
        depth: Option<OwnedPlaneView>,
        normal: Option<OwnedPlaneView>,
        occupancy: Option<OwnedPlaneView>,
        readback_lease: WgpuReadbackLeaseGuard,
    ) -> Self {
        Self {
            selection,
            num_views,
            resolution,
            color,
            depth,
            normal,
            occupancy,
            readback_lease: Some(readback_lease),
        }
    }

    pub(crate) fn zeroed(
        selection: WgpuOutputSelection,
        num_views: usize,
        resolution: usize,
    ) -> Self {
        let pixels_per_view = resolution * resolution;
        let color_bytes_per_view = pixels_per_view * 4;
        Self {
            selection,
            num_views,
            resolution,
            color: selection.color.then(|| {
                OwnedPlaneView::owned_bytes(
                    vec![0; num_views * color_bytes_per_view],
                    color_bytes_per_view,
                    color_bytes_per_view,
                )
            }),
            depth: selection.depth.then(|| {
                OwnedPlaneView::owned_bytes(
                    vec![0; num_views * pixels_per_view],
                    pixels_per_view,
                    pixels_per_view,
                )
            }),
            normal: selection.normal.then(|| {
                let mut bytes = vec![0; num_views * color_bytes_per_view];
                for pixel in bytes.chunks_exact_mut(4) {
                    pixel[3] = 255;
                }
                OwnedPlaneView::owned_bytes(bytes, color_bytes_per_view, color_bytes_per_view)
            }),
            occupancy: selection.occupancy.then(|| {
                OwnedPlaneView::owned_bytes(
                    vec![0; num_views * pixels_per_view],
                    pixels_per_view,
                    pixels_per_view,
                )
            }),
            readback_lease: None,
        }
    }

    pub fn color(&self) -> Option<PlaneView<'_>> {
        self.color.as_ref().map(OwnedPlaneView::as_plane_view)
    }

    pub fn depth(&self) -> Option<PlaneView<'_>> {
        self.depth.as_ref().map(OwnedPlaneView::as_plane_view)
    }

    pub fn normal(&self) -> Option<PlaneView<'_>> {
        self.normal.as_ref().map(OwnedPlaneView::as_plane_view)
    }

    pub fn occupancy(&self) -> Option<PlaneView<'_>> {
        self.occupancy.as_ref().map(OwnedPlaneView::as_plane_view)
    }

    pub fn as_mapped_readback(&self) -> WgpuMappedImageReadback<'_> {
        WgpuMappedImageReadback {
            selection: self.selection,
            num_views: self.num_views,
            resolution: self.resolution,
            color: self.color(),
            depth: self.depth(),
            normal: self.normal(),
            occupancy: self.occupancy(),
        }
    }
}

impl Drop for WgpuOwnedMappedImageReadback {
    fn drop(&mut self) {
        drop(self.color.take());
        drop(self.depth.take());
        drop(self.normal.take());
        drop(self.occupancy.take());
        drop(self.readback_lease.take());
    }
}

impl<'a> WgpuMappedImageReadback<'a> {
    pub fn to_packed_output(&self) -> PackedWgpuImageOutput {
        let pixels_per_view = self.resolution * self.resolution;
        let color_bytes_per_view = pixels_per_view * 4;
        let mut packed = PackedWgpuImageOutput {
            selection: self.selection,
            num_views: self.num_views,
            resolution: self.resolution,
            color: self
                .color
                .map(|_| vec![0; self.num_views * color_bytes_per_view]),
            depth: self
                .depth
                .map(|_| vec![0; self.num_views * pixels_per_view]),
            normal: self
                .normal
                .map(|_| vec![0; self.num_views * color_bytes_per_view]),
            occupancy: self
                .occupancy
                .map(|_| vec![0; self.num_views * pixels_per_view]),
        };

        if let (Some(src), Some(dst)) = (self.color, packed.color.as_mut()) {
            src.copy_views_to(dst, self.num_views);
        }
        if let (Some(src), Some(dst)) = (self.depth, packed.depth.as_mut()) {
            src.copy_views_to(dst, self.num_views);
        }
        if let (Some(src), Some(dst)) = (self.normal, packed.normal.as_mut()) {
            src.copy_views_to(dst, self.num_views);
        }
        if let (Some(src), Some(dst)) = (self.occupancy, packed.occupancy.as_mut()) {
            src.copy_views_to(dst, self.num_views);
        }

        packed
    }

    pub fn to_image_batch(&self) -> ImageBatch {
        self.to_packed_output().into_image_batch()
    }
}

impl PackedWgpuImageOutput {
    pub fn into_image_batch(self) -> ImageBatch {
        let mut image_batch = ImageBatch::new(self.num_views, self.resolution, self.resolution);
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
            let dst_start = view_index * pixels * 4;
            let dst_end = dst_start + pixels * 4;
            images[dst_start..dst_end].copy_from_slice(&color[src_start..src_end]);
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
            let dst_start = view_index * pixels * 4;
            let dst_end = dst_start + pixels * 4;
            normal_maps[dst_start..dst_end].copy_from_slice(&normal[src_start..src_end]);
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
    fn packed_color_output_copies_rgba_to_image_batch() {
        let packed = PackedWgpuImageOutput {
            selection: WgpuOutputSelection::color_only(),
            num_views: 1,
            resolution: 2,
            color: Some(vec![
                10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255,
            ]),
            depth: None,
            normal: None,
            occupancy: None,
        };

        let image_batch = packed.into_image_batch();

        assert_eq!(image_batch.images.shape(), &[1, 2, 2, 4]);
        assert_eq!(
            image_batch.images.as_slice().unwrap(),
            &[10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255]
        );
        assert!(image_batch.depth_maps.iter().all(|value| *value == 0));
        assert!(image_batch
            .normal_maps
            .lanes(ndarray::Axis(3))
            .into_iter()
            .all(|pixel| pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0 && pixel[3] == 255));
        assert!(image_batch.occupancy_maps.iter().all(|value| *value == 0));
    }

    #[test]
    fn packed_partial_maps_leave_unselected_image_batch_maps_zeroed() {
        let packed = PackedWgpuImageOutput {
            selection: WgpuOutputSelection::color_and_occupancy(),
            num_views: 2,
            resolution: 1,
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
        assert!(image_batch
            .normal_maps
            .lanes(ndarray::Axis(3))
            .into_iter()
            .all(|pixel| pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0 && pixel[3] == 255));
    }

    #[test]
    fn mapped_readback_to_packed_output_skips_stride_padding() {
        let color = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 0, 0, 0, 0,
        ];
        let occupancy = vec![41, 42, 43, 44, 0, 0, 0, 0, 45, 46, 47, 48, 0, 0, 0, 0];
        let mapped = WgpuMappedImageReadback {
            selection: WgpuOutputSelection::color_and_occupancy(),
            num_views: 2,
            resolution: 2,
            color: Some(PlaneView {
                bytes: &color,
                bytes_per_view: 16,
                stride_bytes: 20,
            }),
            depth: None,
            normal: None,
            occupancy: Some(PlaneView {
                bytes: &occupancy,
                bytes_per_view: 4,
                stride_bytes: 8,
            }),
        };

        let packed = mapped.to_packed_output();

        assert_eq!(
            packed.color.unwrap(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32
            ]
        );
        assert_eq!(
            packed.occupancy.unwrap(),
            vec![41, 42, 43, 44, 45, 46, 47, 48]
        );
    }

    #[test]
    fn owned_zeroed_readback_uses_rgba_and_scalar_plane_shapes() {
        let readback = WgpuOwnedMappedImageReadback::zeroed(WgpuOutputSelection::all(), 2, 3);

        assert_eq!(readback.num_views, 2);
        assert_eq!(readback.resolution, 3);
        assert_eq!(readback.color().unwrap().bytes_per_view, 3 * 3 * 4);
        assert_eq!(readback.color().unwrap().stride_bytes, 3 * 3 * 4);
        assert_eq!(readback.depth().unwrap().bytes_per_view, 3 * 3);
        assert_eq!(readback.depth().unwrap().stride_bytes, 3 * 3);
        assert_eq!(readback.normal().unwrap().bytes.len(), 2 * 3 * 3 * 4);
        assert!(readback
            .normal()
            .unwrap()
            .bytes
            .chunks_exact(4)
            .all(|pixel| pixel == [0, 0, 0, 255]));
        assert_eq!(readback.occupancy().unwrap().bytes.len(), 2 * 3 * 3);
    }

    #[test]
    fn owned_readback_as_mapped_readback_preserves_plane_strides() {
        let readback = WgpuOwnedMappedImageReadback {
            selection: WgpuOutputSelection::color_and_occupancy(),
            num_views: 2,
            resolution: 2,
            color: Some(OwnedPlaneView::owned_bytes(vec![1; 40], 16, 20)),
            depth: None,
            normal: None,
            occupancy: Some(OwnedPlaneView::owned_bytes(vec![2; 16], 4, 8)),
            readback_lease: None,
        };

        let mapped = readback.as_mapped_readback();

        assert_eq!(mapped.color.unwrap().bytes_per_view, 16);
        assert_eq!(mapped.color.unwrap().stride_bytes, 20);
        assert_eq!(mapped.occupancy.unwrap().bytes_per_view, 4);
        assert_eq!(mapped.occupancy.unwrap().stride_bytes, 8);
        assert!(mapped.depth.is_none());
        assert!(mapped.normal.is_none());
    }
}
