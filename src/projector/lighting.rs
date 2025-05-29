// src/projector/lighting.rs
//! Lighting calculation functions for the projector.
use crate::projector::config::{LightSource, LightingConfig};
use crate::projector::math::{
    dot_product, normalize_vec3, subtract_vec3, transform_point_3d, transform_vector_3d,
};
use crate::projector::mesh::{MeshMaterial, MeshMode};
use ndarray::Array2;

/// Calculates the final linear color based on base color, material properties, lighting, and normal.
pub fn calculate_lighting(
    base_color_linear: [f64; 3], // Base color of the surface (e.g. from texture or vertex color), in linear space
    vertex_position_ws: [f64; 3], // Vertex position in world-space
    normal_ws: [f64; 3],         // Surface normal in world-space
    view_matrix: &Array2<f64>,   // Full 4x4 view matrix that transforms from world to view space
    camera_position_ws: [f64; 3], // Camera position in world-space
    view_dir_ws: [f64; 3],       // View direction in world space (needed for Camera light mode)
    material: &MeshMaterial,     // Material properties of the surface
    mesh_mode: MeshMode,         // Current rendering mode
    lighting_config: &LightingConfig, // Lighting configuration
) -> [f64; 3] {
    if material.illum == 0 && (mesh_mode == MeshMode::ObjOnly || mesh_mode == MeshMode::ObjTexture)
    {
        return base_color_linear;
    }

    // Transform normal and vertex position to view space
    let normal_vs = transform_vector_3d(normal_ws, view_matrix); // Normals are vectors (w=0)
    let vertex_position_vs = transform_point_3d(vertex_position_ws, view_matrix); // Points are transformed (w=1)

    // Determine light direction and view direction in view-space
    let (light_dir_vs, light_intensity_at_surface) = match lighting_config.light_source {
        LightSource::World(world_light_pos) => {
            // Treat world_light_pos as a direction vector in world space and transform it
            let mut dir_vs = transform_vector_3d(world_light_pos, view_matrix);
            normalize_vec3(&mut dir_vs);
            (dir_vs, lighting_config.diffuse_color)
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

            (dir_vs, lighting_config.diffuse_color)
        }
    };

    // View direction in view space is from surface point to camera (origin in VS)
    let mut view_dir_vs = subtract_vec3([0.0, 0.0, 0.0], vertex_position_vs);
    normalize_vec3(&mut view_dir_vs);

    let linear_ka = material.ka;
    let linear_kd = material.kd;
    let linear_ks = material.ks;

    let effective_diffuse_linear = [
        base_color_linear[0] * linear_kd[0],
        base_color_linear[1] * linear_kd[1],
        base_color_linear[2] * linear_kd[2],
    ];

    let mut final_color_linear = [0.0f64; 3];

    let mut normal_n_vs = normal_vs; // Already in view space
    normalize_vec3(&mut normal_n_vs);

    // Ambient component
    let ambient_comp = [
        linear_ka[0] * lighting_config.ambient_color[0],
        linear_ka[1] * lighting_config.ambient_color[1],
        linear_ka[2] * lighting_config.ambient_color[2],
    ];

    // Diffuse component
    let dot_nl = dot_product(&normal_n_vs, &light_dir_vs).max(0.0);
    let diffuse_comp = [
        effective_diffuse_linear[0] * dot_nl * light_intensity_at_surface[0],
        effective_diffuse_linear[1] * dot_nl * light_intensity_at_surface[1],
        effective_diffuse_linear[2] * dot_nl * light_intensity_at_surface[2],
    ];

    // Specular component (Blinn-Phong)
    let mut halfway_dir = [
        light_dir_vs[0] + view_dir_vs[0],
        light_dir_vs[1] + view_dir_vs[1],
        light_dir_vs[2] + view_dir_vs[2],
    ];
    normalize_vec3(&mut halfway_dir);
    let dot_nh = dot_product(&normal_n_vs, &halfway_dir).max(0.0);

    let specular_intensity = if material.ns > 0.0 && lighting_config.shininess > 0.0 {
        dot_nh.powf(lighting_config.shininess)
    } else {
        0.0
    };
    let specular_comp = [
        linear_ks[0] * specular_intensity * lighting_config.specular_color[0],
        linear_ks[1] * specular_intensity * lighting_config.specular_color[1],
        linear_ks[2] * specular_intensity * lighting_config.specular_color[2],
    ];

    match material.illum {
        0 => {
            final_color_linear = [
                base_color_linear[0] * linear_ka[0] * lighting_config.ambient_color[0],
                base_color_linear[1] * linear_ka[1] * lighting_config.ambient_color[1],
                base_color_linear[2] * linear_ka[2] * lighting_config.ambient_color[2],
            ];
        }
        1 => {
            // Diffuse + Ambient
            for i in 0..3 {
                final_color_linear[i] = ambient_comp[i] + diffuse_comp[i];
            }
        }
        _ => {
            // Full model: Ambient + Diffuse + Specular (illum >= 2)
            for i in 0..3 {
                final_color_linear[i] = ambient_comp[i] + diffuse_comp[i] + specular_comp[i];
            }
        }
    }

    final_color_linear[0] = final_color_linear[0].max(0.0).min(1.0);
    final_color_linear[1] = final_color_linear[1].max(0.0).min(1.0);
    final_color_linear[2] = final_color_linear[2].max(0.0).min(1.0);

    final_color_linear
}

// Removed placeholder transform_normal_ws_to_vs as we use transform_vector_3d
