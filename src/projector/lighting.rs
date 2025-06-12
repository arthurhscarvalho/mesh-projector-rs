//! This module provides functions for physically-based lighting calculations,
//! including support for different light sources, material properties, and
//! color space conversions. It is used to compute the final color of surfaces
//! in 3D scenes, taking into account ambient, diffuse, and specular lighting
//! components, as well as transformations between world and view space.

use crate::projector::config::{LightSource, ViewConfig};
use crate::projector::constants::GLOBAL_AMBIENT;
use crate::projector::rasterization::TrianglePoint;
use crate::types::color::Color;
use crate::types::mesh::{MeshMaterial, MeshMode};

/// Calculates the final linear color based on base color, material properties, lighting, and normal.
pub fn calculate_lighting(
    interpolated_point: &TrianglePoint,
    view_config: &ViewConfig,
    material: &MeshMaterial, // Material properties of the surface
    mesh_mode: &MeshMode,    // Current rendering mode
) -> Color {
    // Extract data from interpolated attributes
    let base_color_linear = interpolated_point.base_color;
    let normal_ws = interpolated_point.normal_ws;
    let vertex_position_ws = interpolated_point.position_ws;
    // Extract data from view config
    let transformation_matrix = &view_config.transformation_matrix;
    let light_source = &view_config.light_source;
    let view_dir_ws = view_config.view_direction;
    // For illum model 0, return base color modulated by ambient only
    if material.illum == 0
        && (*mesh_mode == MeshMode::ObjOnly || *mesh_mode == MeshMode::ObjTexture)
    {
        return base_color_linear;
    }
    // Material coefficients are stored in linear space; avoid costly per-pixel gamma conversion
    let linear_ka = Color::from(material.ka);
    let linear_kd = Color::from(material.kd);
    let linear_ks = Color::from(material.ks);
    // Transform normal and vertex position to view space
    let normal_vs = normal_ws.to_view_space_as_vector(transformation_matrix);
    let vertex_position_vs = vertex_position_ws.to_view_space_as_point(transformation_matrix);
    // Normalize the normal in view space
    let mut normal_n_vs = normal_vs;
    normal_n_vs.normalize();
    // Determine light direction and light color in view-space
    let (light_dir_vs, light_color) = match light_source {
        LightSource::World(world_light_pos) => {
            // Treat world_light_pos as a direction vector in world space and transform it
            let mut dir_vs = world_light_pos.to_view_space_as_vector(transformation_matrix);
            dir_vs.normalize();
            // Light direction in view space, and white light
            (dir_vs, Color::white())
        }
        LightSource::Camera => {
            // For orthographic projection with camera lighting, the light direction should
            // match the view direction in world space (transformed to view space)
            // This ensures light comes from the projection plane for any view angle
            let mut dir_ws = view_dir_ws; // Use the passed view direction
            dir_ws.normalize();
            // Transform the world-space view direction to view space
            let mut dir_vs = dir_ws.to_view_space_as_vector(transformation_matrix);
            dir_vs.normalize();
            // Negate to make light point toward surface from camera
            dir_vs = -dir_vs;
            // Light direction in view space, and white light
            (dir_vs, Color::white())
        }
    };
    // View direction in view space is from surface point to camera (origin in VS)
    let mut view_dir_vs = -vertex_position_vs;
    view_dir_vs.normalize();
    // Calculate final color
    let final_color_linear = match material.illum {
        0 => {
            // Constant color - ambient only, modulated by base color
            base_color_linear * linear_ka
        }
        1 => {
            // Ambient component using global ambient and material ambient
            let ambient_comp = GLOBAL_AMBIENT * linear_ka;
            // Diffuse component
            let dot_nl = normal_n_vs.dot(&light_dir_vs).max(0.0);
            let diffuse_comp = base_color_linear * linear_kd * light_color * dot_nl;
            // Ambient + Diffuse
            ambient_comp + diffuse_comp
        }
        _ => {
            // Ambient component using global ambient and material ambient
            let ambient_comp = GLOBAL_AMBIENT * linear_ka;
            // Diffuse component
            let dot_nl = normal_n_vs.dot(&light_dir_vs).max(0.0);
            let diffuse_comp = base_color_linear * linear_kd * light_color * dot_nl;
            // Specular component (Blinn-Phong)
            let mut halfway_dir = light_dir_vs + view_dir_vs;
            halfway_dir.normalize();
            let dot_nh = normal_n_vs.dot(&halfway_dir).max(0.0);
            let specular_intensity = if material.ns > 0.0 {
                dot_nh.powf(material.ns)
            } else {
                0.0
            };
            let specular_comp = linear_ks * light_color * specular_intensity;
            // Full model: Ambient + Diffuse + Specular (illum >= 2)
            ambient_comp + diffuse_comp + specular_comp
        }
    };
    // Clamp to valid range and return the final color
    let final_color_clamped = final_color_linear.clamp(0.0, 1.0);
    final_color_clamped
}
