//! This module implements the core logic for projecting a 3D mesh onto a 2D image plane
//! according to a specified view configuration. It orchestrates the rasterization of mesh
//! triangles, manages per-pixel buffers (color, depth, normals, occupancy), and applies
//! post-processing such as outlier filtering.

use crate::projector::config::ProjectionConfig;
use crate::projector::constants::{IMAGE_CHANNELS, NORMAL_CHANNELS};
use crate::renderer::buffers::ImageBuffers;
use crate::renderer::config::RenderPassConfig;
use crate::renderer::filtering::filter_outliers;
use crate::renderer::msaa::{sample_offsets, MsaaAccumulator};
use crate::renderer::rasterization::rasterize_triangle_to_images;
use crate::types::mesh::Scene;
use ndarray::{Array2, Array3};

/// Computes a single orthographic projection for the given view configuration.
pub fn compute_projection(
    mesh: &Scene,
    settings: &ProjectionConfig,
    render_pass_config: &RenderPassConfig,
) -> (Array3<u8>, Array2<u8>, Array3<u8>, Array2<u8>, u64) {
    let max_bound: u16 = settings.resolution;
    let rows = max_bound as usize;
    let cols = max_bound as usize;
    let background_color = settings.background.to_srgba8();
    let num_submeshes = mesh.meshes.len();
    let sample_count = settings.msaa.sample_count.samples();
    let sample_offsets = sample_offsets(sample_count);
    let mut msaa_accum =
        MsaaAccumulator::new(rows, cols, IMAGE_CHANNELS, NORMAL_CHANNELS, background_color);
    for &(off_x, off_y) in &sample_offsets {
        let mut buffers = ImageBuffers::new(
            rows,
            cols,
            IMAGE_CHANNELS,
            NORMAL_CHANNELS,
            background_color,
        );
        // Rasterize all triangles from all meshes
        for submesh_index in 0..num_submeshes {
            let num_faces = mesh.meshes[submesh_index].geometry.triangles.nrows();
            for triangle_index in 0..num_faces {
                rasterize_triangle_to_images(
                    &mesh,
                    render_pass_config,
                    &mut buffers,
                    submesh_index,
                    triangle_index, // Local triangle index for geometry
                    (off_x, off_y),
                );
            }
        }
        msaa_accum.accumulate(&buffers);
    }
    let mut buffers = msaa_accum.resolve(sample_count);
    // Apply outlier filtering
    filter_outliers(&mut buffers, &settings, max_bound as f32);
    // Flip buffers
    buffers.v_flip();
    buffers.h_flip();
    // Return from buffers
    (
        buffers.image,
        buffers.depth_map,
        buffers.normal_map,
        buffers.occupancy_map,
        buffers.outlier_count,
    )
}
