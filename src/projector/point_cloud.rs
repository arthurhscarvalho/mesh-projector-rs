//! This module contains the functions for discretizing a mesh into a 3D point cloud.
//! Instead of projecting the mesh onto a 2D plane, this function samples points
//! across the triangle surfaces in 3D space.

use crate::projector::config::{LightSource, PointCloudConfig, ViewConfig};
use crate::projector::geometry::create_view_matrix;
use crate::projector::{geometry, rasterization};
use crate::types::coordinate::Coordinate;
use crate::types::mesh::Mesh;
use crate::types::orientation::Orientation;
use ndarray::Array2;

/// Discretizes a mesh into a 3D point cloud, maintaining the 3D structure.
/// Instead of projecting the mesh onto a 2D plane, this function samples points
/// across the triangle surfaces in 3D space.
///
/// Returns:
/// - points: Array2<f64> - 3D points in the range [0, max_bound]
/// - colors: Array2<u8> - RGB colors for each point (0-255 range)
pub fn discretize_mesh_to_point_cloud(
    mesh: &Mesh,
    settings: &PointCloudConfig,
) -> (Array2<f64>, Array2<u8>) {
    log::info!(
        "Initializing 3D mesh discretization. Mode: {:?}, Precision: {}",
        mesh.mode,
        settings.precision
    );
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let density_factor = settings.point_cloud_density_factor;
    // Initialize point cloud buffers
    let mut points = Vec::new();
    let mut colors = Vec::new();
    let mut num_points: usize = 0;
    // Calculate total mesh area for density calculation
    let mut total_mesh_area = 0.0;
    for submesh_index in 0..mesh.submeshes.len() {
        let submesh = &mesh.submeshes[submesh_index];
        let num_faces = submesh.geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            let triangle_data =
                rasterization::extract_triangle_data(&mesh, submesh_index, triangle_index);
            let triangle_area = geometry::calculate_triangle_area(&triangle_data.vertices_ws);
            total_mesh_area += triangle_area;
        }
    }
    // Density factor is the number of points per unit area.
    let density = (density_factor * max_bound_f64.powf(2.0)) / total_mesh_area.max(1e-9);
    // Define a transformation matrix for the world coordinates
    let angles = Orientation::new(0.0, 0.0, 0.0);
    let rotation_matrix = angles.to_rotation_matrix();
    let transformation_matrix = create_view_matrix(&rotation_matrix);
    // Define the view direction
    let view_direction = Coordinate::new(
        rotation_matrix[[0, 2]],
        rotation_matrix[[1, 2]],
        rotation_matrix[[2, 2]],
    );
    // Define the light source
    let light_source = LightSource::World((1.0, 1.0, 1.0).into());
    // Create a view config for lighting calculations
    let view_config = ViewConfig {
        view_direction,
        transformation_matrix,
        max_bound_f64,
        light_source,
    };
    // Process each submesh
    for submesh_index in 0..mesh.submeshes.len() {
        let submesh = &mesh.submeshes[submesh_index];
        let num_faces = submesh.geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            // Extract triangle data
            let triangle_data =
                rasterization::extract_triangle_data(&mesh, submesh_index, triangle_index);
            // Calculate triangle area
            let triangle_area = geometry::calculate_triangle_area(&triangle_data.vertices_ws);
            // Calculate number of subdivisions based on triangle area and density
            // We need to ensure that the number of subdivisions is at least 1
            let n_subdiv = (triangle_area * density).sqrt().floor().max(1.0) as usize;
            // Sample points across the triangle surface
            for i in 0..=n_subdiv {
                for j in 0..=(n_subdiv - i) {
                    // Calculate barycentric coordinates
                    let u = i as f64 / n_subdiv as f64;
                    let v = j as f64 / n_subdiv as f64;
                    let w = 1.0 - u - v;
                    if u < 0.0 || v < 0.0 || w < 0.0 {
                        continue;
                    }
                    // Use the shared interpolation function
                    let attributes = rasterization::interpolate_triangle_attributes(
                        &triangle_data,
                        (u, v, w),
                        mesh,
                    );
                    // Calculate final color using the shared lighting function
                    let final_color = rasterization::calculate_final_color(
                        &attributes,
                        &view_config,
                        &submesh.material,
                        mesh.mode,
                    )
                    .to_srgb_denormalized();
                    // Add point and color to the flatten arrays
                    points.push(attributes.position_ws.x());
                    points.push(attributes.position_ws.y());
                    points.push(attributes.position_ws.z());
                    colors.push(final_color.r() as u8);
                    colors.push(final_color.g() as u8);
                    colors.push(final_color.b() as u8);
                    num_points += 1;
                }
            }
        }
    }
    let points_array = nd::Array2::from_shape_vec((num_points, 3), points).unwrap();
    let colors_array = nd::Array2::from_shape_vec((num_points, 3), colors).unwrap();
    (points_array, colors_array)
}
