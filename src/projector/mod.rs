// src/projector/mod.rs
mod buffers;
pub mod config;
mod constants;
mod filtering;
pub mod geometry;
mod lighting;
mod math;
pub mod mesh;
mod rasterization;
mod sampling;

use crate::mesh_io::save_image_data;
use crate::projector::buffers::Buffers;
use crate::projector::config::{ProjectorConfig, ViewConfig};
use crate::projector::constants::{DEFAULT_MISSING_TEXTURE_COLOR, NUM_CHANNELS};
use crate::projector::filtering::filter_outliers;
use crate::projector::math::{
    combine_rotations, normalize_vec3, rotation_matrix_x, rotation_matrix_y, rotation_matrix_z,
};
use mesh::{Mesh, MeshMode};
use ndarray::{array, s, Array2, Array3, Array4, Axis};
use rasterization::rasterize_triangle;
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

/// Creates a 4x4 view matrix from a 3x3 rotation matrix.
/// For orthographic projection, the camera is the projection plane itself.
/// This creates a view matrix that transforms from world space to view space,
/// where the view direction becomes the Z axis.
fn create_view_matrix(rotation_matrix: &Array2<f64>) -> Array2<f64> {
    // In orthographic projection, we don't position the camera at a distance
    // Instead, the projection plane itself acts as the camera
    // The rotation matrix should already be set up to transform the view direction to the Z axis
    let r = rotation_matrix;

    // Create the 4x4 view matrix without translation component
    // The rotation part ensures the view direction is transformed to the Z axis
    array![
        [r[[0, 0]], r[[0, 1]], r[[0, 2]], 0.0],
        [r[[1, 0]], r[[1, 1]], r[[1, 2]], 0.0],
        [r[[2, 0]], r[[2, 1]], r[[2, 2]], 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
}

/// Computes a single orthographic projection for the given view configuration.
fn compute_single_projection(
    mesh: &Mesh,
    settings: &ProjectorConfig,
    view_config: &ViewConfig,
) -> (Array4<u8>, Array4<u8>, Array4<u8>, Array3<u8>, u64) {
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;

    // Initialize buffers for a single view
    let mut buffers = Buffers::new(1, rows, cols, NUM_CHANNELS);

    let mut image_view = buffers.images.index_axis_mut(Axis(0), 0);
    let mut depth_map_view = buffers.depth_maps.index_axis_mut(Axis(0), 0);
    let mut normal_map_view = buffers.normal_maps.index_axis_mut(Axis(0), 0);
    let mut z_buffer_view = buffers.z_buffers.index_axis_mut(Axis(0), 0);

    let mut global_triangle_offset = 0;
    // Rasterize all triangles from all submeshes
    for submesh in &mesh.submeshes {
        let num_faces = submesh.geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            // triangle_index is local to the submesh
            let face_normal_view = submesh.geometry.face_normals.row(triangle_index);
            let uv_lookup_triangle_id = global_triangle_offset + triangle_index;
            rasterize_triangle(
                &submesh.geometry,
                &submesh.material,
                &mesh.texture,
                mesh.mode,
                triangle_index,        // Local triangle index for geometry
                uv_lookup_triangle_id, // Global triangle ID for UVs
                face_normal_view,
                view_config,
                &mut z_buffer_view.view_mut(),
                &mut image_view.view_mut(),
                &mut depth_map_view.view_mut(),
                &mut normal_map_view.view_mut(),
            );
        }
        global_triangle_offset += num_faces;
    }

    // Apply outlier filtering
    let (_, outlier_count) = filter_outliers(
        &mut buffers.images,
        &buffers.z_buffers,
        settings.filtering_window_radius,
        max_bound_f64,
    );

    // Create occupancy map
    let mut occupancy_map = Array3::<u8>::zeros((1, rows, cols));
    let z_slice = buffers.z_buffers.index_axis(Axis(0), 0);
    for r in 0..rows {
        for c in 0..cols {
            let depth_val = z_slice[[r, c]];
            let is_occupied = depth_val < f64::INFINITY && depth_val.is_finite();
            if is_occupied {
                occupancy_map[[0, r, c]] = 255;
            }
        }
    }

    // Create new 4D arrays with the correct shape
    let mut single_image = Array4::<u8>::zeros((1, rows, cols, NUM_CHANNELS));
    let mut single_depth = Array4::<u8>::zeros((1, rows, cols, NUM_CHANNELS));
    let mut single_normal = Array4::<u8>::zeros((1, rows, cols, NUM_CHANNELS));

    // Copy data from the buffers
    single_image
        .slice_mut(s![0, .., .., ..])
        .assign(&buffers.images.slice(s![0, .., .., ..]));
    single_depth
        .slice_mut(s![0, .., .., ..])
        .assign(&buffers.depth_maps.slice(s![0, .., .., ..]));
    single_normal
        .slice_mut(s![0, .., .., ..])
        .assign(&buffers.normal_maps.slice(s![0, .., .., ..]));

    (
        single_image,
        single_depth,
        single_normal,
        occupancy_map,
        outlier_count[0],
    )
}

/// Computes projections based on the specified mode (either 6 orthographic views or rotation-based).
///
/// Returns:
/// Tuple: (images, depth_maps, normal_maps, occupancy_maps, outlier_counts)
/// where occupancy_maps are RGB images (255 for occupied, 0 for empty).
pub fn compute_mesh_projections(
    mesh: &Mesh,
    settings: &ProjectorConfig,
) -> (Array4<u8>, Array4<u8>, Array4<u8>, Array3<u8>, Vec<u64>) {
    log::info!(
        "Initializing projector. Mode: {:?}, Precision: {}, Filtering: {}",
        mesh.mode,
        settings.precision,
        settings.filtering_window_radius
    );

    // Setup & Validation
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;

    // Handle empty meshes
    if mesh.submeshes.is_empty()
        || mesh
            .submeshes
            .iter()
            .all(|sm| sm.geometry.vertices.is_empty() || sm.geometry.triangles.is_empty())
    {
        log::warn!(
            "Empty mesh (no submeshes or no geometry in submeshes). Returning empty buffers."
        );
        let empty_buffers = Buffers::new(1, rows, cols, NUM_CHANNELS);
        let empty_occupancy_map = Array3::<u8>::zeros((1, rows, cols));
        return (
            empty_buffers.images,
            empty_buffers.depth_maps,
            empty_buffers.normal_maps,
            empty_occupancy_map,
            vec![0],
        );
    }

    // Determine number of views and prepare view configurations
    let rotation_config = &settings.rotation_config;

    let num_views = rotation_config.num_steps;
    let mut view_configs = Vec::with_capacity(num_views);

    // Create view configs for rotation-based projection
    for i in 0..num_views {
        let angle_x = rotation_config.angle_delta_x * i as f64;
        let angle_y = rotation_config.angle_delta_y * i as f64;
        let angle_z = rotation_config.angle_delta_z * i as f64;

        let rx = rotation_matrix_x(angle_x);
        let ry = rotation_matrix_y(angle_y);
        let rz = rotation_matrix_z(angle_z);
        let rotation_matrix = combine_rotations(&rx, &ry, &rz);

        let view_dir_tuple = (
            rotation_matrix[[0, 2]],
            rotation_matrix[[1, 2]],
            rotation_matrix[[2, 2]],
        );

        view_configs.push(ViewConfig {
            view_dir: view_dir_tuple,
            rotation_matrix: rotation_matrix.clone(),
            view_matrix: create_view_matrix(&rotation_matrix),
            max_bound_f64,
            light_source: config::LightSource::Camera,
        });
    }

    let settings_arc = Arc::new(settings);
    let mesh_arc = Arc::new(mesh.clone());

    let mut projections = Array4::<u8>::zeros((num_views, rows, cols, NUM_CHANNELS));
    let mut depth_maps = Array4::<u8>::zeros((num_views, rows, cols, NUM_CHANNELS));
    let mut normal_maps = Array4::<u8>::zeros((num_views, rows, cols, NUM_CHANNELS));
    let mut occupancy_maps = Array3::<u8>::zeros((num_views, rows, cols));
    let mut outlier_counts = vec![0; num_views];

    // Compute projections for each view in parallel
    projections
        .axis_iter_mut(Axis(0))
        .zip(depth_maps.axis_iter_mut(Axis(0)))
        .zip(normal_maps.axis_iter_mut(Axis(0)))
        .zip(occupancy_maps.axis_iter_mut(Axis(0)))
        .zip(outlier_counts.iter_mut())
        .collect::<Vec<_>>()
        .into_par_iter()
        .enumerate()
        .for_each(
            |(
                i,
                (
                    (((mut projection, mut depth_map), mut normal_map), mut occupancy_map),
                    outlier_count,
                ),
            )| {
                let view_config = &view_configs[i];
                log::info!("Computing projection {} of {}", i + 1, num_views);

                // Compute projection for this view
                let (img, depth, normal, occ, outlier) =
                    compute_single_projection(&mesh_arc, &settings_arc, view_config);

                // Store results in the 5D array and other containers
                projection.assign(&img);
                depth_map.assign(&depth);
                normal_map.assign(&normal);
                occupancy_map.assign(&occ);
                *outlier_count = outlier;
            },
        );

    (
        projections,
        depth_maps,
        normal_maps,
        occupancy_maps,
        outlier_counts,
    )
}

/// Computes rotation-based projections and saves them on-the-fly in parallel.
/// This function does not return any projection data.
pub fn compute_and_save_rotation_projections(
    mesh: &Mesh,
    settings: &ProjectorConfig,
    output_dir: &Path,
) {
    log::info!(
        "Initializing rotation-based projector. Precision: {}, Filtering: {}",
        settings.precision,
        settings.filtering_window_radius
    );

    let rotation_config = &settings.rotation_config;

    // Setup & Validation
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;

    // Handle empty meshes - CORRECTED
    if mesh.submeshes.is_empty()
        || mesh
            .submeshes
            .iter()
            .all(|sm| sm.geometry.vertices.is_empty() || sm.geometry.triangles.is_empty())
    {
        log::warn!("Empty mesh (no submeshes or no geometry in submeshes). Skipping projection.");
        return;
    }

    let num_steps = rotation_config.num_steps;
    let mut view_configs = Vec::with_capacity(num_steps);

    // Create view configs for rotation-based projection
    for i in 0..num_steps {
        let angle_x = rotation_config.angle_delta_x * i as f64;
        let angle_y = rotation_config.angle_delta_y * i as f64;
        let angle_z = rotation_config.angle_delta_z * i as f64;

        let rx = rotation_matrix_x(angle_x);
        let ry = rotation_matrix_y(angle_y);
        let rz = rotation_matrix_z(angle_z);
        let rotation_matrix = combine_rotations(&rx, &ry, &rz);

        let view_dir_tuple = (
            rotation_matrix[[0, 2]],
            rotation_matrix[[1, 2]],
            rotation_matrix[[2, 2]],
        );

        view_configs.push(ViewConfig {
            view_dir: view_dir_tuple,
            rotation_matrix: rotation_matrix.clone(),
            view_matrix: create_view_matrix(&rotation_matrix),
            max_bound_f64,
            light_source: config::LightSource::Camera,
        });
    }

    let mesh_arc = Arc::new(mesh.clone());
    let settings_arc = Arc::new(settings);
    let output_dir_arc = Arc::new(output_dir.to_path_buf());

    view_configs
        .par_iter()
        .enumerate()
        .for_each(|(i, view_config)| {
            log::info!("Computing projection {} of {}", i + 1, num_steps);

            // Mesh is no longer rotated here. Transformation happens in compute_single_projection.
            let current_mesh_for_projection = Arc::clone(&mesh_arc).as_ref().clone();

            let (img, _, _, _, _) =
                compute_single_projection(&current_mesh_for_projection, &settings_arc, view_config);

            // Save the projection data for the current view
            if let Err(e) = save_image_data(&img, &output_dir_arc, i, true) {
                log::error!("Failed to save data for view {}: {}", i, e);
            }
        });

    log::info!("Finished computing and saving all rotation-based projections.");
}

/// Discretizes a mesh into a 3D point cloud, maintaining the 3D structure.
/// Instead of projecting the mesh onto a 2D plane, this function samples points
/// across the triangle surfaces in 3D space.
///
/// Returns:
/// - points: Array2<f64> - 3D points in the range [0, max_bound]
/// - colors: Array2<u8> - RGB colors for each point (0-255 range)
pub fn discretize_mesh_to_point_cloud(
    mesh: &Mesh,
    settings: &ProjectorConfig,
) -> (ndarray::Array2<f64>, ndarray::Array2<u8>) {
    log::info!(
        "Initializing 3D mesh discretization. Mode: {:?}, Precision: {}",
        mesh.mode,
        settings.precision
    );
    let max_bound: u64 = 1 << settings.precision;
    let max_bound_f64: f64 = max_bound as f64;

    if mesh.submeshes.is_empty()
        || mesh
            .submeshes
            .iter()
            .all(|sm| sm.geometry.vertices.is_empty() || sm.geometry.triangles.is_empty())
    {
        log::warn!(
            "Empty mesh (no submeshes or no geometry in submeshes). Returning empty point cloud."
        );
        let empty_points = ndarray::Array2::<f64>::zeros((0, 3));
        let empty_colors = ndarray::Array2::<u8>::zeros((0, 3));
        return (empty_points, empty_colors);
    }

    let mut point_cloud_points = Vec::new();
    let mut point_cloud_colors = Vec::new();

    let mut total_mesh_area = 0.0;
    for submesh in &mesh.submeshes {
        let triangles = submesh.geometry.triangles.view();
        let world_vertices = submesh.geometry.vertices.view(); // These are now assumed to be pre-scaled
        for triangle_index in 0..triangles.nrows() {
            let tri_indices = triangles.row(triangle_index);
            let i0 = tri_indices[0];
            let i1 = tri_indices[1];
            let i2 = tri_indices[2];
            let p0 = world_vertices.row(i0);
            let p1 = world_vertices.row(i1);
            let p2 = world_vertices.row(i2);
            let ab = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
            let ac = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let area =
                0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            total_mesh_area += area;
        }
    }

    // Density calculation remains, using max_bound_f64 and total_mesh_area (calculated from pre-scaled verts)
    let density = (settings.point_cloud_density_factor * max_bound_f64 * max_bound_f64)
        / total_mesh_area.max(1e-9);

    let mut global_triangle_offset = 0;
    for submesh in &mesh.submeshes {
        let triangles = submesh.geometry.triangles.view();
        let world_vertices = submesh.geometry.vertices.view(); // Assumed pre-scaled
        let num_faces = triangles.nrows();
        for triangle_index in 0..num_faces {
            let face_normal_view = submesh.geometry.face_normals.row(triangle_index);
            let face_normal_ws_arr = [
                face_normal_view[0],
                face_normal_view[1],
                face_normal_view[2],
            ];
            let tri_indices = triangles.row(triangle_index);
            let i0 = tri_indices[0];
            let i1 = tri_indices[1];
            let i2 = tri_indices[2];
            let p0_ws_vec = world_vertices.row(i0);
            let p1_ws_vec = world_vertices.row(i1);
            let p2_ws_vec = world_vertices.row(i2);
            let p0_ws_arr = [p0_ws_vec[0], p0_ws_vec[1], p0_ws_vec[2]];
            let p1_ws_arr = [p1_ws_vec[0], p1_ws_vec[1], p1_ws_vec[2]];
            let p2_ws_arr = [p2_ws_vec[0], p2_ws_vec[1], p2_ws_vec[2]];
            let mut n0_ws_arr = face_normal_ws_arr;
            let mut n1_ws_arr = face_normal_ws_arr;
            let mut n2_ws_arr = face_normal_ws_arr;
            if let Some(v_normals_ws) = &submesh.geometry.vertex_normals {
                let v_normals_ws_view = v_normals_ws.view();
                let n0_v_ws = v_normals_ws_view.row(i0);
                let n1_v_ws = v_normals_ws_view.row(i1);
                let n2_v_ws = v_normals_ws_view.row(i2);
                n0_ws_arr = [n0_v_ws[0], n0_v_ws[1], n0_v_ws[2]];
                n1_ws_arr = [n1_v_ws[0], n1_v_ws[1], n1_v_ws[2]];
                n2_ws_arr = [n2_v_ws[0], n2_v_ws[1], n2_v_ws[2]];
            }
            let mut uv0 = (0.0, 0.0);
            let mut uv1 = (0.0, 0.0);
            let mut uv2 = (0.0, 0.0);
            let uv_triangle_id = global_triangle_offset + triangle_index;
            if mesh.mode == MeshMode::Full || mesh.mode == MeshMode::ObjTexture {
                if let Some(uvs) = &mesh.texture.triangle_uvs {
                    let uv_idx_base = uv_triangle_id * 3;
                    if uv_idx_base + 2 < uvs.shape()[0] {
                        let uv0_data = uvs.row(uv_idx_base);
                        let uv1_data = uvs.row(uv_idx_base + 1);
                        let uv2_data = uvs.row(uv_idx_base + 2);
                        uv0 = (uv0_data[0], uv0_data[1]);
                        uv1 = (uv1_data[0], uv1_data[1]);
                        uv2 = (uv2_data[0], uv2_data[1]);
                    }
                }
            }
            let mut c0 = (1.0, 1.0, 1.0);
            let mut c1 = (1.0, 1.0, 1.0);
            let mut c2 = (1.0, 1.0, 1.0);
            if mesh.mode == MeshMode::ObjOnly {
                if let Some(v_colors) = &submesh.geometry.vertex_colors {
                    if i0 < v_colors.nrows() && i1 < v_colors.nrows() && i2 < v_colors.nrows() {
                        let v_colors_view = v_colors.view();
                        let c0v = v_colors_view.row(i0);
                        let c1v = v_colors_view.row(i1);
                        let c2v = v_colors_view.row(i2);
                        c0 = (c0v[0].powf(2.2), c0v[1].powf(2.2), c0v[2].powf(2.2));
                        c1 = (c1v[0].powf(2.2), c1v[1].powf(2.2), c1v[2].powf(2.2));
                        c2 = (c2v[0].powf(2.2), c2v[1].powf(2.2), c2v[2].powf(2.2));
                    }
                }
            }
            let ab = [
                p1_ws_arr[0] - p0_ws_arr[0],
                p1_ws_arr[1] - p0_ws_arr[1],
                p1_ws_arr[2] - p0_ws_arr[2],
            ];
            let ac = [
                p2_ws_arr[0] - p0_ws_arr[0],
                p2_ws_arr[1] - p0_ws_arr[1],
                p2_ws_arr[2] - p0_ws_arr[2],
            ];
            let cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let triangle_area =
                0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();

            // n_subdiv determines subdivisions along triangle edges.
            // Number of points per triangle is roughly 0.5 * n_subdiv^2.
            // With n_subdiv = (triangle_area * density).sqrt(),
            // points per triangle becomes ~ 0.5 * triangle_area * density.
            // Using .floor() allows n_subdiv to be 0 for triangles that don't meet
            // the threshold for even one subdivision step at the current density,
            // leading to fewer points at low precision settings.
            let n_subdiv = (triangle_area * density).sqrt().floor() as usize;

            if n_subdiv == 0 {
                continue;
            }
            for i in 0..=n_subdiv {
                for j in 0..=(n_subdiv - i) {
                    let u = i as f64 / n_subdiv as f64;
                    let v = j as f64 / n_subdiv as f64;
                    let w = 1.0 - u - v;
                    if u < 0.0 || v < 0.0 || w < 0.0 {
                        continue;
                    }
                    let x = u * p0_ws_arr[0] + v * p1_ws_arr[0] + w * p2_ws_arr[0];
                    let y = u * p0_ws_arr[1] + v * p1_ws_arr[1] + w * p2_ws_arr[1];
                    let z = u * p0_ws_arr[2] + v * p1_ws_arr[2] + w * p2_ws_arr[2];
                    let mut interpolated_normal_ws = [
                        u * n0_ws_arr[0] + v * n1_ws_arr[0] + w * n2_ws_arr[0],
                        u * n0_ws_arr[1] + v * n1_ws_arr[1] + w * n2_ws_arr[1],
                        u * n0_ws_arr[2] + v * n1_ws_arr[2] + w * n2_ws_arr[2],
                    ];
                    normalize_vec3(&mut interpolated_normal_ws);
                    let (interpolated_u, interpolated_v) =
                        if mesh.mode == MeshMode::Full || mesh.mode == MeshMode::ObjTexture {
                            (
                                u * uv0.0 + v * uv1.0 + w * uv2.0,
                                u * uv0.1 + v * uv1.1 + w * uv2.1,
                            )
                        } else {
                            (0.0, 0.0)
                        };
                    let linear_base_color = match mesh.mode {
                        MeshMode::ObjOnly => {
                            let r = u * c0.0 + v * c1.0 + w * c2.0;
                            let g = u * c0.1 + v * c1.1 + w * c2.1;
                            let b = u * c0.2 + v * c1.2 + w * c2.2;
                            [r.max(0.0), g.max(0.0), b.max(0.0)]
                        }
                        MeshMode::ObjTexture | MeshMode::Full => {
                            if let Some(tex_image) = &mesh.texture.texture {
                                let sampled_srgb = sampling::sample_texture_color_bilinear(
                                    interpolated_u,
                                    interpolated_v,
                                    tex_image.view(),
                                    mesh.texture.channels,
                                    mesh.texture.height_f64,
                                    mesh.texture.width_f64,
                                );
                                [
                                    sampled_srgb[0].powf(2.2),
                                    sampled_srgb[1].powf(2.2),
                                    sampled_srgb[2].powf(2.2),
                                ]
                            } else {
                                DEFAULT_MISSING_TEXTURE_COLOR
                            }
                        }
                    };
                    let final_color = lighting::calculate_lighting(
                        linear_base_color,
                        [x, y, z],
                        interpolated_normal_ws,
                        &ndarray::Array2::eye(4),
                        [0.0, 0.0, 1.0],
                        &submesh.material,
                        mesh.mode,
                        &config::LightSource::Camera,
                    );
                    let r_srgb = (final_color[0].powf(1.0 / 2.2) * 255.0).min(255.0) as u8;
                    let g_srgb = (final_color[1].powf(1.0 / 2.2) * 255.0).min(255.0) as u8;
                    let b_srgb = (final_color[2].powf(1.0 / 2.2) * 255.0).min(255.0) as u8;

                    // Points are used directly as they are assumed to be in the target [0, max_bound_f64] range.
                    // Scaling step is removed:
                    // let scaled_x = (x - min_x) / (max_x - min_x) * max_bound_f64;
                    // let scaled_y = (y - min_y) / (max_y - min_y) * max_bound_f64;
                    // let scaled_z = (z - min_z) / (max_z - min_z) * max_bound_f64;
                    // point_cloud_points.push([scaled_x, scaled_y, scaled_z]);
                    point_cloud_points.push([x, y, z]); // Use x, y, z directly

                    point_cloud_colors.push([r_srgb, g_srgb, b_srgb]);
                }
            }
        }
        global_triangle_offset += num_faces;
    }
    let num_points = point_cloud_points.len();
    let mut points_array = ndarray::Array2::<f64>::zeros((num_points, 3));
    let mut colors_array = ndarray::Array2::<u8>::zeros((num_points, 3));
    for (i, point) in point_cloud_points.iter().enumerate() {
        points_array[[i, 0]] = point[0];
        points_array[[i, 1]] = point[1];
        points_array[[i, 2]] = point[2];
    }
    for (i, color) in point_cloud_colors.iter().enumerate() {
        colors_array[[i, 0]] = color[0];
        colors_array[[i, 1]] = color[1];
        colors_array[[i, 2]] = color[2];
    }
    (points_array, colors_array)
}
