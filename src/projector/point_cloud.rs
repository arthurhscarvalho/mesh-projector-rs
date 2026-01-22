//! This module contains the functions for discretizing a mesh into a 3D point cloud.
//! Instead of projecting the mesh onto a 2D plane, this function samples points
//! across the triangle surfaces in 3D space.

use crate::projector::config::PointCloudConfig;
use crate::renderer::buffers::PointCloudBuffers;
use crate::renderer::config::{
    LightingConfig, ProjectionModel, RenderPassConfig, SsaoConfig, ViewConfig,
};
use crate::renderer::geometry::create_view_matrix;
use crate::renderer::rasterization::rasterize_triangle_to_point_cloud;
use crate::types::coordinate::Coordinate;
use crate::types::mesh::Scene;
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
    mesh: &Scene,
    settings: &PointCloudConfig,
    lighting_config: &LightingConfig,
) -> (Array2<f32>, Array2<u8>) {
    log::info!(
        "Initializing 3D mesh discretization. Mode: {:?}, Resolution: {}",
        mesh.mode,
        settings.resolution
    );
    let max_bound_f32 = settings.resolution as f32;
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
    // Create view and lighting configs for lighting calculations
    let view_config = ViewConfig {
        view_direction,
        transformation_matrix,
        max_bound_f32,
        projection: ProjectionModel::Orthographic,
    };
    let ssao_config = SsaoConfig::disabled();
    let render_pass_config = RenderPassConfig {
        view: view_config,
        lighting: lighting_config,
        ssao: &ssao_config,
    };
    // Initialize point cloud buffers
    let mut buffers = PointCloudBuffers::new();
    // Process each submesh
    for submesh_index in 0..mesh.meshes.len() {
        let submesh = &mesh.meshes[submesh_index];
        let num_faces = submesh.geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            rasterize_triangle_to_point_cloud(
                mesh,
                settings,
                &render_pass_config,
                &mut buffers,
                submesh_index,
                triangle_index,
            );
        }
    }
    buffers.to_array()
}
