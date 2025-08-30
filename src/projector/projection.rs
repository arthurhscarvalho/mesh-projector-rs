//! This module implements the core logic for projecting a 3D mesh onto a 2D image plane
//! according to a specified view configuration. It orchestrates the rasterization of mesh
//! triangles, manages per-pixel buffers (color, depth, normals, occupancy), and applies
//! post-processing such as outlier filtering.

use crate::projector::config::ProjectionConfig;
use crate::projector::constants::NUM_CHANNELS;
use crate::renderer::buffers::ImageBuffers;
use crate::renderer::config::ViewConfig;
use crate::renderer::filtering::filter_outliers;
use crate::renderer::rasterization::rasterize_triangle_to_images;
use crate::types::mesh::Scene;
use ndarray::{Array2, Array3};

/// Computes a single orthographic projection for the given view configuration.
pub fn compute_projection(
    mesh: &Scene,
    settings: &ProjectionConfig,
    view_config: &ViewConfig,
) -> (Array3<u8>, Array2<u8>, Array3<u8>, Array2<u8>, u64) {
    let max_bound: u64 = 1 << settings.precision;
    let rows = max_bound as usize;
    let cols = max_bound as usize;
    // Initialize buffers for a single view
    let mut buffers = ImageBuffers::new(rows, cols, NUM_CHANNELS);
    let num_submeshes = mesh.meshes.len();
    // Rasterize all triangles from all meshes
    for submesh_index in 0..num_submeshes {
        let num_faces = mesh.meshes[submesh_index].geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            rasterize_triangle_to_images(
                &mesh,
                view_config,
                &mut buffers,
                submesh_index,
                triangle_index, // Local triangle index for geometry
            );
        }
    }
    // Apply outlier filtering
    filter_outliers(&mut buffers, &settings, max_bound as f32);
    // Vertically flip buffers
    buffers.v_flip();
    // Return from buffers
    (
        buffers.image,
        buffers.depth_map,
        buffers.normal_map,
        buffers.occupancy_map,
        buffers.outlier_count,
    )
}
