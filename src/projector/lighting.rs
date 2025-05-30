// src/projector/lighting.rs
//! Lighting calculation functions for the projector.
use crate::projector::config::LightSource;
use crate::projector::constants::GLOBAL_AMBIENT;
use crate::projector::math::{
    dot_product, normalize_vec3, subtract_vec3, transform_point_3d, transform_vector_3d,
};
use crate::projector::mesh::{MeshMaterial, MeshMode};
use ndarray::Array2;

/// Converts sRGB color to linear space
fn srgb_to_linear(srgb: [f64; 3]) -> [f64; 3] {
    [srgb[0].powf(2.2), srgb[1].powf(2.2), srgb[2].powf(2.2)]
}

/// Calculates the final linear color based on base color, material properties, lighting, and normal.
pub fn calculate_lighting(
    base_color_linear: [f64; 3], // Base color of the surface (e.g. from texture or vertex color), in linear space
    vertex_position_ws: [f64; 3], // Vertex position in world-space
    normal_ws: [f64; 3],         // Surface normal in world-space
    view_matrix: &Array2<f64>,   // Full 4x4 view matrix that transforms from world to view space
    view_dir_ws: [f64; 3],       // View direction in world space (needed for Camera light mode)
    material: &MeshMaterial,     // Material properties of the surface
    mesh_mode: MeshMode,         // Current rendering mode
    light_source: &LightSource,  // Lighting configuration
) -> [f64; 3] {
    // For illum model 0, return base color modulated by ambient only
    if material.illum == 0 && (mesh_mode == MeshMode::ObjOnly || mesh_mode == MeshMode::ObjTexture)
    {
        return base_color_linear;
    }

    // Convert material properties from sRGB to linear space
    let linear_ka = srgb_to_linear(material.ka);
    let linear_kd = srgb_to_linear(material.kd);
    let linear_ks = srgb_to_linear(material.ks);

    // Transform normal and vertex position to view space
    let normal_vs = transform_vector_3d(normal_ws, view_matrix); // Normals are vectors (w=0)
    let vertex_position_vs = transform_point_3d(vertex_position_ws, view_matrix); // Points are transformed (w=1)

    // Normalize the normal in view space
    let mut normal_n_vs = normal_vs;
    normalize_vec3(&mut normal_n_vs);

    // Determine light direction and light color in view-space
    let (light_dir_vs, light_color) = match light_source {
        LightSource::World(world_light_pos) => {
            // Treat world_light_pos as a direction vector in world space and transform it
            let mut dir_vs = transform_vector_3d(*world_light_pos, view_matrix);
            normalize_vec3(&mut dir_vs);
            (dir_vs, [1.0, 1.0, 1.0]) // White light
        }
        LightSource::Camera => {
            // For orthographic projection with camera lighting, the light direction should
            // match the view direction in world space (transformed to view space)
            // This ensures light comes from the projection plane for any view angle
            let mut dir_ws = view_dir_ws; // Use the passed view direction
            normalize_vec3(&mut dir_ws);

            // Transform the world-space view direction to view space
            let mut dir_vs = transform_vector_3d(dir_ws, view_matrix);
            normalize_vec3(&mut dir_vs);

            // Negate to make light point toward surface from camera
            dir_vs = [-dir_vs[0], -dir_vs[1], -dir_vs[2]];

            (dir_vs, [1.0, 1.0, 1.0]) // White light
        }
    };

    // View direction in view space is from surface point to camera (origin in VS)
    let mut view_dir_vs = subtract_vec3([0.0, 0.0, 0.0], vertex_position_vs);
    normalize_vec3(&mut view_dir_vs);

    let mut final_color_linear = match material.illum {
        0 => {
            // Constant color - ambient only, modulated by base color
            [
                base_color_linear[0] * linear_ka[0],
                base_color_linear[1] * linear_ka[1],
                base_color_linear[2] * linear_ka[2],
            ]
        }
        1 => {
            // Ambient + Diffuse

            // Ambient component using global ambient and material ambient
            let ambient_comp = [
                GLOBAL_AMBIENT[0] * linear_ka[0],
                GLOBAL_AMBIENT[1] * linear_ka[1],
                GLOBAL_AMBIENT[2] * linear_ka[2],
            ];

            // Diffuse component
            let dot_nl = dot_product(&normal_n_vs, &light_dir_vs).max(0.0);
            let diffuse_comp = [
                base_color_linear[0] * linear_kd[0] * dot_nl * light_color[0],
                base_color_linear[1] * linear_kd[1] * dot_nl * light_color[1],
                base_color_linear[2] * linear_kd[2] * dot_nl * light_color[2],
            ];

            [
                ambient_comp[0] + diffuse_comp[0],
                ambient_comp[1] + diffuse_comp[1],
                ambient_comp[2] + diffuse_comp[2],
            ]
        }
        _ => {
            // Full model: Ambient + Diffuse + Specular (illum >= 2)

            // Ambient component using global ambient and material ambient
            let ambient_comp = [
                GLOBAL_AMBIENT[0] * linear_ka[0],
                GLOBAL_AMBIENT[1] * linear_ka[1],
                GLOBAL_AMBIENT[2] * linear_ka[2],
            ];

            // Diffuse component
            let dot_nl = dot_product(&normal_n_vs, &light_dir_vs).max(0.0);
            let diffuse_comp = [
                base_color_linear[0] * linear_kd[0] * dot_nl * light_color[0],
                base_color_linear[1] * linear_kd[1] * dot_nl * light_color[1],
                base_color_linear[2] * linear_kd[2] * dot_nl * light_color[2],
            ];

            // Specular component (Blinn-Phong)
            let mut halfway_dir = [
                light_dir_vs[0] + view_dir_vs[0],
                light_dir_vs[1] + view_dir_vs[1],
                light_dir_vs[2] + view_dir_vs[2],
            ];
            normalize_vec3(&mut halfway_dir);
            let dot_nh = dot_product(&normal_n_vs, &halfway_dir).max(0.0);

            let specular_intensity = if material.ns > 0.0 {
                dot_nh.powf(material.ns)
            } else {
                0.0
            };

            let specular_comp = [
                linear_ks[0] * specular_intensity * light_color[0],
                linear_ks[1] * specular_intensity * light_color[1],
                linear_ks[2] * specular_intensity * light_color[2],
            ];

            [
                ambient_comp[0] + diffuse_comp[0] + specular_comp[0],
                ambient_comp[1] + diffuse_comp[1] + specular_comp[1],
                ambient_comp[2] + diffuse_comp[2] + specular_comp[2],
            ]
        }
    };

    // Clamp to valid range
    final_color_linear[0] = final_color_linear[0].max(0.0).min(1.0);
    final_color_linear[1] = final_color_linear[1].max(0.0).min(1.0);
    final_color_linear[2] = final_color_linear[2].max(0.0).min(1.0);

    final_color_linear
}

// Removed placeholder transform_normal_ws_to_vs as we use transform_vector_3d
