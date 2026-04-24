use crate::image::output::{OwnedPlaneView, PlaneView};
use crate::point_cloud::resources::PointCloudReadbackLeaseGuard;
use ndarray::Array2;

#[derive(Clone, Copy, Debug)]
pub struct WgpuMappedPointCloudReadback<'a> {
    pub num_points: usize,
    pub positions: PlaneView<'a>,
    pub colors: PlaneView<'a>,
}

pub struct WgpuOwnedMappedPointCloudReadback {
    pub num_points: usize,
    positions: OwnedPlaneView,
    colors: OwnedPlaneView,
    #[allow(dead_code)]
    readback_lease: Option<PointCloudReadbackLeaseGuard>,
}

impl WgpuOwnedMappedPointCloudReadback {
    pub(crate) fn new(
        num_points: usize,
        positions: OwnedPlaneView,
        colors: OwnedPlaneView,
        readback_lease: PointCloudReadbackLeaseGuard,
    ) -> Self {
        Self {
            num_points,
            positions,
            colors,
            readback_lease: Some(readback_lease),
        }
    }

    pub(crate) fn zeroed() -> Self {
        Self {
            num_points: 0,
            positions: OwnedPlaneView::owned_bytes(Vec::new(), 0, std::mem::size_of::<[f32; 4]>()),
            colors: OwnedPlaneView::owned_bytes(Vec::new(), 0, std::mem::size_of::<u32>()),
            readback_lease: None,
        }
    }

    pub fn positions(&self) -> PlaneView<'_> {
        self.positions.as_plane_view()
    }

    pub fn colors(&self) -> PlaneView<'_> {
        self.colors.as_plane_view()
    }

    pub fn as_mapped_readback(&self) -> WgpuMappedPointCloudReadback<'_> {
        WgpuMappedPointCloudReadback {
            num_points: self.num_points,
            positions: self.positions(),
            colors: self.colors(),
        }
    }

    pub fn to_arrays(&self) -> (Array2<f32>, Array2<u8>) {
        self.as_mapped_readback().to_arrays()
    }
}

impl<'a> WgpuMappedPointCloudReadback<'a> {
    pub fn to_arrays(&self) -> (Array2<f32>, Array2<u8>) {
        if self.num_points == 0 {
            return (Array2::zeros((0, 3)), Array2::zeros((0, 3)));
        }

        let positions_padded: &[[f32; 4]] = bytemuck::cast_slice(self.positions.bytes);
        let colors_packed: &[u32] = bytemuck::cast_slice(self.colors.bytes);
        let mut points = Vec::with_capacity(self.num_points * 3);
        let mut colors = Vec::with_capacity(self.num_points * 3);

        for point in positions_padded.iter().take(self.num_points) {
            points.extend_from_slice(&point[0..3]);
        }
        for packed in colors_packed.iter().take(self.num_points) {
            colors.push((packed & 255) as u8);
            colors.push(((packed >> 8) & 255) as u8);
            colors.push(((packed >> 16) & 255) as u8);
        }

        (
            Array2::from_shape_vec((self.num_points, 3), points)
                .expect("point cloud positions shape is valid"),
            Array2::from_shape_vec((self.num_points, 3), colors)
                .expect("point cloud colors shape is valid"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::output::OwnedPlaneView;

    #[test]
    fn mapped_point_cloud_readback_converts_padded_outputs_to_arrays() {
        let positions = [
            1.0f32.to_ne_bytes(),
            2.0f32.to_ne_bytes(),
            3.0f32.to_ne_bytes(),
            0.0f32.to_ne_bytes(),
            4.0f32.to_ne_bytes(),
            5.0f32.to_ne_bytes(),
            6.0f32.to_ne_bytes(),
            0.0f32.to_ne_bytes(),
        ]
        .concat();
        let colors = vec![10, 20, 30, 0, 40, 50, 60, 0];
        let mapped = WgpuMappedPointCloudReadback {
            num_points: 2,
            positions: PlaneView {
                bytes: &positions,
                bytes_per_view: positions.len(),
                stride_bytes: std::mem::size_of::<[f32; 4]>(),
            },
            colors: PlaneView {
                bytes: &colors,
                bytes_per_view: colors.len(),
                stride_bytes: std::mem::size_of::<u32>(),
            },
        };

        let (points, colors) = mapped.to_arrays();

        assert_eq!(points.shape(), &[2, 3]);
        assert_eq!(colors.shape(), &[2, 3]);
        assert_eq!(points.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(colors.as_slice().unwrap(), &[10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn owned_point_cloud_readback_preserves_plane_metadata() {
        let readback = WgpuOwnedMappedPointCloudReadback {
            num_points: 2,
            positions: OwnedPlaneView::owned_bytes(
                vec![0; 2 * std::mem::size_of::<[f32; 4]>()],
                2 * std::mem::size_of::<[f32; 4]>(),
                std::mem::size_of::<[f32; 4]>(),
            ),
            colors: OwnedPlaneView::owned_bytes(
                vec![0; 2 * std::mem::size_of::<u32>()],
                2 * std::mem::size_of::<u32>(),
                std::mem::size_of::<u32>(),
            ),
            readback_lease: None,
        };

        assert_eq!(readback.positions().bytes_per_view, 32);
        assert_eq!(readback.positions().stride_bytes, 16);
        assert_eq!(readback.colors().bytes_per_view, 8);
        assert_eq!(readback.colors().stride_bytes, 4);
    }
}
