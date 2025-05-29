// src/projector/lighting.rs
//! Lighting calculation functions for the projector.
use crate::projector::constants::GLOBAL_AMBIENT;
use crate::projector::math::{dot_product, normalize_vec3, reflect};
use crate::projector::mesh::{MeshMaterial, MeshMode};

/// Calculates the final linear color based on base color, material properties, lighting, and normal.
pub fn calculate_lighting(
    base_color_linear: [f64; 3], // Base color of the surface (e.g. from texture or vertex color), in linear space
    normal_vs: [f64; 3],         // Surface normal in view-space
    light_dir_vs: [f64; 3],      // Light direction in view-space (points towards the light source)
    view_pos_vs: [f64; 3],       // Camera/view position in view-space
    material: &MeshMaterial,     // Material properties of the surface
    mesh_mode: MeshMode,         // Current rendering mode
    light_color: [f64; 3],       // Color/intensity of the light source
) -> [f64; 3] {
    // If mesh_mode is ObjOnly or ObjTexture, and illum is 0, treat as unlit (color = base_color)
    // Illum 0: Color on and Ambient off. This typically means color = diffuse_color * base_color, no lighting.
    // For our setup: if illum is 0, it implies the base_color is the final color.
    if material.illum == 0 && (mesh_mode == MeshMode::ObjOnly || mesh_mode == MeshMode::ObjTexture)
    {
        return base_color_linear;
    }

    let linear_ka = material.ka; // Assume Ka, Kd, Ks from material are already linear or sRGB that we treat as linear here.
    let linear_kd = material.kd;
    let linear_ks = material.ks;

    // Effective diffuse color after modulation by base_color (e.g., texture)
    let effective_diffuse_linear = [
        base_color_linear[0] * linear_kd[0],
        base_color_linear[1] * linear_kd[1],
        base_color_linear[2] * linear_kd[2],
    ];

    let mut final_color_linear = [0.0f64; 3];

    // Normalize necessary vectors
    let mut normal_n = normal_vs;
    normalize_vec3(&mut normal_n);
    let mut light_dir_n = light_dir_vs;
    normalize_vec3(&mut light_dir_n);

    // View direction: from surface point to camera. Here, view_pos_vs is camera position.
    // For orthographic, view direction is constant, e.g. [0,0,1] if camera looks along -Z.
    // For simplicity, let's assume view_dir is always [0,0,1] in view space for Blinn-Phong.
    // More accurately, it would be normalize(view_pos_vs - surface_point_vs), but for a distant viewer/ortho, it's constant.
    let view_dir_n = [0.0, 0.0, 1.0]; // Standard view direction in view space

    match material.illum {
        0 => {
            // Color = Kd. No lighting. Our base_color_linear already incorporates Kd implicitly via effective_diffuse_linear.
            // This case is handled by the check at the beginning for ObjOnly/ObjTexture modes.
            // For Full mode with illum 0, it implies using the diffuse color as is.
            final_color_linear = effective_diffuse_linear;
        }
        1 => {
            // Ambient + Diffuse
            let ambient_term = [
                linear_ka[0] * GLOBAL_AMBIENT[0],
                linear_ka[1] * GLOBAL_AMBIENT[1],
                linear_ka[2] * GLOBAL_AMBIENT[2],
            ];
            let dot_nl = dot_product(&normal_n, &light_dir_n).max(0.0);
            let diffuse_term = [
                effective_diffuse_linear[0] * dot_nl * light_color[0],
                effective_diffuse_linear[1] * dot_nl * light_color[1],
                effective_diffuse_linear[2] * dot_nl * light_color[2],
            ];
            for i in 0..3 {
                final_color_linear[i] = ambient_term[i] + diffuse_term[i];
            }
        }
        2 | _ => {
            // Phong/Blinn-Phong based model
            let ambient_comp = [
                linear_ka[0] * GLOBAL_AMBIENT[0],
                linear_ka[1] * GLOBAL_AMBIENT[1],
                linear_ka[2] * GLOBAL_AMBIENT[2],
            ];

            let dot_nl = dot_product(&normal_n, &light_dir_n).max(0.0);
            let diffuse_comp = [
                effective_diffuse_linear[0] * dot_nl * light_color[0],
                effective_diffuse_linear[1] * dot_nl * light_color[1],
                effective_diffuse_linear[2] * dot_nl * light_color[2],
            ];

            // Blinn-Phong: Halfway vector
            let mut halfway_dir = [
                light_dir_n[0] + view_dir_n[0],
                light_dir_n[1] + view_dir_n[1],
                light_dir_n[2] + view_dir_n[2],
            ];
            normalize_vec3(&mut halfway_dir);
            let dot_nh = dot_product(&normal_n, &halfway_dir).max(0.0);

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

            for i in 0..3 {
                final_color_linear[i] = ambient_comp[i] + diffuse_comp[i] + specular_comp[i];
            }
        }
    }

    // Clamp final color components to avoid negative values, though physically based rendering might allow them transiently.
    final_color_linear[0] = final_color_linear[0].max(0.0);
    final_color_linear[1] = final_color_linear[1].max(0.0);
    final_color_linear[2] = final_color_linear[2].max(0.0);

    // Apply opacity - not typically done in lighting, but rather in compositing/blending.
    // If opacity is meant to affect light transmission for this surface, it's more complex.
    // For now, opacity is handled by the rasterizer if it leads to discarding pixels or alpha blending.
    // Here, we just return the calculated color of the surface itself.

    final_color_linear
}
