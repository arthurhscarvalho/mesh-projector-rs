// src/projector/rasterization.rs
use crate::projector::config::ProjectionConfig;
use crate::projector::constants::{DEFAULT_LIGHT_COLOR, DEFAULT_MISSING_TEXTURE_COLOR};
use crate::projector::geometry::{
    barycentric, calculate_bounding_box, is_inside_triangle, is_triangle_visible,
};
use crate::projector::lighting::calculate_lighting;
use crate::projector::math::normalize_vec3;
use crate::projector::mesh::Mesh;
use crate::projector::sampling::sample_texture_color_bilinear;
use crate::projector::texture::TextureData;
use crate::ProjectionMode;
use nd::{Array2, ArrayViewMut2, ArrayViewMut3};
use std::f64;

/// Rasterizes a single triangle based on the projection mode.
pub fn rasterize_triangle(
    t_idx: usize,                           // Index of the current triangle
    triangle_normals: &Array2<f64>, // Triangle face normals (M, 3) - Precomputed or provided
    mesh: &Mesh,                    // Mesh with material properties
    texture_data: Option<&TextureData<'_>>, // Optional diffuse texture data
    projection_config: &ProjectionConfig, // Configuration for the current projection view
    z_slice: &mut ArrayViewMut2<f64>, // Mutable view to the Z-buffer slice
    image_slice: &mut ArrayViewMut3<u8>, // Mutable view to the color image buffer slice
    mode: ProjectionMode,           // Rendering mode
) {
    // Back-face Culling
    let geometric_normal_view = triangle_normals.row(t_idx);
    let face_normal = [
        geometric_normal_view[0],
        geometric_normal_view[1],
        geometric_normal_view[2],
    ];
    if !is_triangle_visible(
        face_normal[0],
        face_normal[1],
        face_normal[2],
        projection_config.view_dir,
    ) {
        return;
    }
    // Get Vertex Data Indices & Positions
    let points = mesh.vertices.view();
    let triangles = mesh.triangles.view();
    let tri_indices = triangles.row(t_idx);
    let i0 = tri_indices[0];
    let i1 = tri_indices[1];
    let i2 = tri_indices[2];
    let p0 = points.row(i0);
    let p1 = points.row(i1);
    let p2 = points.row(i2);
    // Setup Data for Interpolation
    let z0 = p0[projection_config.depth_axis];
    let z1 = p1[projection_config.depth_axis];
    let z2 = p2[projection_config.depth_axis];
    // Store raw attributes for linear interpolation
    let mut n0 = (0.0, 0.0, 0.0);
    let mut n1 = (0.0, 0.0, 0.0);
    let mut n2 = (0.0, 0.0, 0.0);
    let mut uv0 = (0.0, 0.0);
    let mut uv1 = (0.0, 0.0);
    let mut uv2 = (0.0, 0.0);
    // Default vertex color is white (linear)
    let mut c0 = (1.0, 1.0, 1.0);
    let mut c1 = (1.0, 1.0, 1.0);
    let mut c2 = (1.0, 1.0, 1.0);
    let mut has_normals = false;
    // Setup normals if available
    if let Some(v_normals) = &mesh.vertex_normals {
        let v_normals_view = v_normals.view();
        let n0v = v_normals_view.row(i0);
        let n1v = v_normals_view.row(i1);
        let n2v = v_normals_view.row(i2);
        // Store raw normals
        n0 = (n0v[0], n0v[1], n0v[2]);
        n1 = (n1v[0], n1v[1], n1v[2]);
        n2 = (n2v[0], n2v[1], n2v[2]);
        has_normals = true;
    }
    // Setup UVs if available (required for texture modes)
    if mode == ProjectionMode::Full || mode == ProjectionMode::ObjTexture {
        if let Some(uvs) = &mesh.triangle_uvs {
            let uvs_view = uvs.view();
            let uv_idx_base = t_idx * 3;
            let uv0_data = uvs_view.row(uv_idx_base);
            let uv1_data = uvs_view.row(uv_idx_base + 1);
            let uv2_data = uvs_view.row(uv_idx_base + 2);
            // Store raw UVs
            uv0 = (uv0_data[0], uv0_data[1]);
            uv1 = (uv1_data[0], uv1_data[1]);
            uv2 = (uv2_data[0], uv2_data[1]);
        } else {
            // Default UVs are (0,0)
            uv0 = (0.0, 0.0);
            uv1 = (0.0, 0.0);
            uv2 = (0.0, 0.0);
        }
    }
    // Setup vertex colors if available (required for ObjOnly mode)
    if mode == ProjectionMode::ObjOnly {
        if let Some(v_colors) = &mesh.vertex_colors {
            let v_colors_view = v_colors.view();
            let c0v = v_colors_view.row(i0);
            let c1v = v_colors_view.row(i1);
            let c2v = v_colors_view.row(i2);
            // Assume vertex colors are sRGB [0,1], store linear for interpolation
            c0 = (c0v[0].powf(2.2), c0v[1].powf(2.2), c0v[2].powf(2.2));
            c1 = (c1v[0].powf(2.2), c1v[1].powf(2.2), c1v[2].powf(2.2));
            c2 = (c2v[0].powf(2.2), c2v[1].powf(2.2), c2v[2].powf(2.2));
        }
        // If None, the default white setup (1,1,1) remains.
    }
    // Project Vertices to 2D
    let p0_proj = (p0[projection_config.p_axis], p0[projection_config.q_axis]);
    let p1_proj = (p1[projection_config.p_axis], p1[projection_config.q_axis]);
    let p2_proj = (p2[projection_config.p_axis], p2[projection_config.q_axis]);

    // Calculate Bounding Box
    let (rows, cols) = (z_slice.shape()[0], z_slice.shape()[1]);
    let Some((min_row, max_row, min_col, max_col)) =
        calculate_bounding_box(p0_proj, p1_proj, p2_proj, rows, cols)
    else {
        return;
    };
    // Pixel Loop
    for row in min_row..max_row {
        for col in min_col..max_col {
            let pixel_center = (col as f64 + 0.5, row as f64 + 0.5);
            // Barycentric Coordinates
            let Some((w0, w1, w2)) = barycentric(p0_proj, p1_proj, p2_proj, pixel_center) else {
                continue;
            };
            // Check if inside using Barycentric coordinates
            if is_inside_triangle(w0, w1, w2) {
                // Interpolate depth linearly
                let interpolated_z = w0 * z0 + w1 * z1 + w2 * z2;
                // Depth Test
                let current_z = z_slice[[row, col]];
                let depth_test_passed = if projection_config.is_front_view {
                    interpolated_z < current_z
                } else {
                    interpolated_z > current_z
                };
                if depth_test_passed {
                    // Interpolate Attributes
                    let interpolated_normal = if has_normals {
                        let bw0 = w0;
                        let bw1 = w1;
                        let bw2 = w2;
                        // Linear interpolation of normals
                        let nx = bw0 * n0.0 + bw1 * n1.0 + bw2 * n2.0;
                        let ny = bw0 * n0.1 + bw1 * n1.1 + bw2 * n2.1;
                        let nz = bw0 * n0.2 + bw1 * n1.2 + bw2 * n2.2;
                        let mut norm = [nx, ny, nz];
                        normalize_vec3(&mut norm);
                        norm
                    } else {
                        face_normal
                    };
                    let (interpolated_u, interpolated_v) =
                        if mode == ProjectionMode::Full || mode == ProjectionMode::ObjTexture {
                            // Linear interpolation of UVs
                            let u = w0 * uv0.0 + w1 * uv1.0 + w2 * uv2.0;
                            let v = w0 * uv0.1 + w1 * uv1.1 + w2 * uv2.1;
                            (u, v)
                        } else {
                            (0.0, 0.0)
                        };
                    let linear_base_color = match mode {
                        ProjectionMode::ObjOnly => {
                            // Linear interpolation of vertex colors (already in linear space)
                            let r = w0 * c0.0 + w1 * c1.0 + w2 * c2.0;
                            let g = w0 * c0.1 + w1 * c1.1 + w2 * c2.1;
                            let b = w0 * c0.2 + w1 * c1.2 + w2 * c2.2;
                            [r.max(0.0), g.max(0.0), b.max(0.0)]
                        }
                        ProjectionMode::ObjTexture | ProjectionMode::Full => {
                            if let Some(tex) = texture_data {
                                let sampled_srgb = sample_texture_color_bilinear(
                                    interpolated_u,
                                    interpolated_v,
                                    tex,
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
                    // Calculate Final Color (Lighting)
                    let view_dir_tuple = projection_config.view_dir;
                    let view_dir_array = [view_dir_tuple.0, view_dir_tuple.1, view_dir_tuple.2];
                    let light_dir_array =
                        [-view_dir_array[0], -view_dir_array[1], -view_dir_array[2]];
                    let light_color = DEFAULT_LIGHT_COLOR;
                    let final_linear_color = calculate_lighting(
                        linear_base_color,
                        interpolated_normal,
                        view_dir_array,
                        light_dir_array,
                        light_color,
                        &mesh.ka,
                        &mesh.kd,
                        &mesh.ks,
                        mesh.ns,
                        mesh.illum,
                    );
                    // Alpha Blending
                    let final_color_to_write_linear = if mesh.opacity < 0.999 {
                        let alpha = mesh.opacity;
                        // Read existing color from buffer (sRGB u8)
                        let old_r_u8 = image_slice[[row, col, 0]];
                        let old_g_u8 = image_slice[[row, col, 1]];
                        let old_b_u8 = image_slice[[row, col, 2]];
                        // Convert existing color to linear f64
                        let old_r_linear = (old_r_u8 as f64 / 255.0).powf(2.2);
                        let old_g_linear = (old_g_u8 as f64 / 255.0).powf(2.2);
                        let old_b_linear = (old_b_u8 as f64 / 255.0).powf(2.2);
                        // Perform alpha blending in linear space
                        let blended_r =
                            final_linear_color[0] * alpha + old_r_linear * (1.0 - alpha);
                        let blended_g =
                            final_linear_color[1] * alpha + old_g_linear * (1.0 - alpha);
                        let blended_b =
                            final_linear_color[2] * alpha + old_b_linear * (1.0 - alpha);
                        [blended_r, blended_g, blended_b]
                    } else {
                        // Opaque, use calculated color directly
                        final_linear_color
                    };
                    // Gamma Correction (Linear -> sRGB) & Clamping
                    let final_color_srgb = [
                        final_color_to_write_linear[0]
                            .powf(1.0 / 2.2)
                            .max(0.0)
                            .min(1.0),
                        final_color_to_write_linear[1]
                            .powf(1.0 / 2.2)
                            .max(0.0)
                            .min(1.0),
                        final_color_to_write_linear[2]
                            .powf(1.0 / 2.2)
                            .max(0.0)
                            .min(1.0),
                    ];
                    // Update Buffers
                    z_slice[[row, col]] = interpolated_z;
                    image_slice[[row, col, 0]] = (final_color_srgb[0] * 255.0) as u8;
                    image_slice[[row, col, 1]] = (final_color_srgb[1] * 255.0) as u8;
                    image_slice[[row, col, 2]] = (final_color_srgb[2] * 255.0) as u8;
                }
            }
        }
    }
}
