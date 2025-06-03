use crate::projector::buffers::Buffers;
use crate::projector::config::{ProjectorConfig, ViewConfig};
use crate::projector::constants::NUM_CHANNELS;
use crate::projector::filtering::filter_outliers;
use crate::projector::mesh::Mesh;
use crate::projector::rasterization::rasterize_triangle;
use ndarray::{Array2, Array3};

/// Computes a single orthographic projection for the given view configuration.
pub fn compute_projection(
    mesh: &Mesh,
    settings: &ProjectorConfig,
    view_config: &ViewConfig,
) -> (Array3<u8>, Array3<u8>, Array3<u8>, Array2<u8>, u64) {
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;
    // Initialize buffers for a single view
    let mut buffers = Buffers::new(rows, cols, NUM_CHANNELS);
    let num_submeshes = mesh.submeshes.len();
    // Rasterize all triangles from all submeshes
    for submesh_index in 0..num_submeshes {
        let num_faces = mesh.submeshes[submesh_index].geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            rasterize_triangle(
                &mesh,
                view_config,
                &mut buffers,
                submesh_index,
                triangle_index, // Local triangle index for geometry
            );
        }
    }
    // Create occupancy map
    for r in 0..rows {
        for c in 0..cols {
            let depth_val = buffers.z_buffer[[r, c]];
            let is_occupied = depth_val < f64::INFINITY && depth_val.is_finite();
            if is_occupied {
                buffers.occupancy_map[[r, c]] = 255;
            }
        }
    }
    // Apply outlier filtering
    filter_outliers(
        &mut buffers,
        settings.filtering_window_radius,
        max_bound_f64,
    );
    // Return from buffers
    (
        buffers.image,
        buffers.depth_map,
        buffers.normal_map,
        buffers.occupancy_map,
        buffers.outlier_count,
    )
}
