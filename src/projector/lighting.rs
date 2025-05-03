// src/projector/lighting.rs
//! Lighting calculation functions for the projector.

use crate::projector::constants::GLOBAL_AMBIENT;
use crate::projector::math::{dot_product, normalize_vec3, reflect};

/// Calculates the final linear color based on base color, material properties, lighting, and normal.
pub fn calculate_lighting(
    linear_base_color: [f64; 3],
    normal: [f64; 3],
    view_dir: [f64; 3],
    light_dir: [f64; 3],
    light_color: [f64; 3],
    ka: &[f64; 3],
    kd: &[f64; 3],
    ks: &[f64; 3],
    ns: f64,
    illum: i64,
) -> [f64; 3] {
    let linear_ka = [ka[0].powf(2.2), ka[1].powf(2.2), ka[2].powf(2.2)];
    let linear_kd = [kd[0].powf(2.2), kd[1].powf(2.2), kd[2].powf(2.2)];
    let linear_ks = [ks[0].powf(2.2), ks[1].powf(2.2), ks[2].powf(2.2)];

    let effective_diffuse_linear = [
        linear_base_color[0] * linear_kd[0],
        linear_base_color[1] * linear_kd[1],
        linear_base_color[2] * linear_kd[2],
    ];

    let mut final_color_linear = [0.0f64; 3];
    match illum {
        0 => {
            // Base color only
            final_color_linear = linear_base_color;
        }
        1 => {
            // Ambient + Diffuse (Original simple model)
            let ambient_term = linear_ka;
            let dot_nl = dot_product(&normal, &light_dir).max(0.0);
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
            // Illum 2+ (Revised Phong model variation)
            // Ambient term: Modulate Ka by a global ambient factor
            let ambient_term = [
                linear_ka[0] * GLOBAL_AMBIENT[0],
                linear_ka[1] * GLOBAL_AMBIENT[1],
                linear_ka[2] * GLOBAL_AMBIENT[2],
            ];

            // Diffuse term (without light color initially)
            let dot_nl = dot_product(&normal, &light_dir).max(0.0);
            let diffuse_term = [
                effective_diffuse_linear[0] * dot_nl,
                effective_diffuse_linear[1] * dot_nl,
                effective_diffuse_linear[2] * dot_nl,
            ];

            // Specular term (without light color initially)
            let mut reflected_light_dir = [0.0; 3];
            let neg_light_dir = [-light_dir[0], -light_dir[1], -light_dir[2]];
            reflect(&neg_light_dir, &normal, &mut reflected_light_dir);
            normalize_vec3(&mut reflected_light_dir);
            let dot_rv = dot_product(&reflected_light_dir, &view_dir).max(0.0);
            let specular_intensity = if ns > 0.0 { dot_rv.powf(ns) } else { 0.0 };
            let specular_term = [
                linear_ks[0] * specular_intensity,
                linear_ks[1] * specular_intensity,
                linear_ks[2] * specular_intensity,
            ];

            // Combine: Final = Ambient + LightColor * (Diffuse + Specular)
            for i in 0..3 {
                final_color_linear[i] =
                    ambient_term[i] + light_color[i] * (diffuse_term[i] + specular_term[i]);
            }
        }
    }
    final_color_linear
}
