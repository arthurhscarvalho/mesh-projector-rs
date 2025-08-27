//! This module contains the functions for discretizing a mesh into a 3D point cloud.
//! Instead of projecting the mesh onto a 2D plane, this function samples points
//! across the triangle surfaces in 3D space.

use crate::projector::config::PointCloudConfig;
use crate::renderer::buffers::PointCloudBuffers;
use crate::renderer::config::{LightSource, ViewConfig};
use crate::renderer::geometry::create_view_matrix;
use crate::renderer::rasterization::rasterize_triangle_to_point_cloud;
use crate::types::coordinate::Coordinate;
use crate::types::mesh::Mesh;
use crate::types::orientation::Orientation;
use ndarray::Array2;

/// Discretizes a mesh into a 3D point cloud, maintaining the 3D structure.
/// Instead of projecting the mesh onto a 2D plane, this function samples points
/// across the triangle surfaces in 3D space.
///
/// Returns:
/// - points: Array2<f32> - 3D points in the range [0, max_bound]
/// - colors: Array2<u8> - RGB colors for each point (0-255 range)
pub fn discretize_mesh_to_point_cloud(
    mesh: &Mesh,
    settings: &PointCloudConfig,
) -> (Array2<f32>, Array2<u8>) {
    log::info!(
        "Initializing 3D mesh discretization. Mode: {:?}, Precision: {}",
        mesh.mode,
        settings.precision
    );
    let max_bound_f32 = (1 << settings.precision) as f32;
    // Define a transformation matrix for the world coordinates
    let angles = Orientation::new(0.0, 0.0, 0.0);
    let rotation_matrix = angles.to_rotation_matrix();
    let transformation_matrix = create_view_matrix(&rotation_matrix);
    // The view direction is set to the third column of the rotation matrix,
    // which corresponds to the camera's forward (Z) axis in view space.
    let view_direction = Coordinate::new(
        rotation_matrix[[0, 2]],
        rotation_matrix[[1, 2]],
        rotation_matrix[[2, 2]],
    );
    // Define the light source
    let light_source = LightSource::Directional(Coordinate::new(0.0, -1.0, 0.0));
    // Create a view config for lighting calculations
    let view_config = ViewConfig {
        view_direction,
        transformation_matrix,
        max_bound_f32,
        light_source,
    };
    // Initialize point cloud buffers
    let mut buffers = PointCloudBuffers::new();
    // Process each submesh
    for submesh_index in 0..mesh.submeshes.len() {
        let submesh = &mesh.submeshes[submesh_index];
        let num_faces = submesh.geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            rasterize_triangle_to_point_cloud(
                mesh,
                settings,
                &view_config,
                &mut buffers,
                submesh_index,
                triangle_index,
            );
        }
    }
    buffers.to_array()
}
