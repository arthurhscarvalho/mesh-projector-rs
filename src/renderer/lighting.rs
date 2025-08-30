//! This module provides functions for physically-based lighting calculations,
//! including support for different light sources, material properties, and
//! color space conversions. It is used to compute the final color of surfaces
//! in 3D scenes, taking into account ambient, diffuse, and specular lighting
//! components, as well as transformations between world and view space.

use crate::renderer::config::{LightSource, ViewConfig};
use crate::renderer::rasterization::TrianglePoint;
use crate::types::color::Color;
use crate::types::coordinate::Coordinate;
use crate::types::mesh::{MeshMaterial, MeshMode};

fn calculate_light_direction(view_config: &ViewConfig, normal_n_vs: Coordinate) -> Coordinate {
    let light_source = &view_config.light_source;
    let view_dir_ws = view_config.view_direction;
    let transformation_matrix = &view_config.transformation_matrix;
    let light_dir_vs = match light_source {
        LightSource::Ambient => {
            // Ambient lighting: no specific light direction. We emulate uniform illumination
            // by treating the light as coming from all directions equally. To achieve this,
            // we set the light direction equal to the surface normal (so the diffuse term
            // evaluates to 1) and use white light. This effectively disables directional
            // shading while still preserving material interactions (Ka/Kd/Ks).
            normal_n_vs
        }
        LightSource::Directional(view_light_dir) => {
            let mut dir_vs = *view_light_dir;
            dir_vs.normalize();
            dir_vs
        }
        LightSource::Headlight => {
            // For orthographic projection with camera lighting, the light direction should
            // match the view direction in world space (transformed to view space)
            // This ensures light comes from the projection plane for any view angle
            let mut dir_ws = view_dir_ws;
            dir_ws.normalize();
            // Transform the world-space view direction to view space
            let mut dir_vs = dir_ws.to_view_space_as_vector(transformation_matrix);
            dir_vs.normalize();
            // Negate to make light point toward surface from camera
            dir_vs = -dir_vs;
            // Light direction in view space, and white light
            dir_vs
        }
        LightSource::Point(world_light_pos) => {
            // Treat world_light_pos as a direction vector in world space and transform it
            let mut dir_vs = world_light_pos.to_view_space_as_vector(transformation_matrix);
            dir_vs.normalize();
            // Light direction in view space, and white light
            dir_vs
        }
    };
    light_dir_vs
}

/// Calculates the final linear color based on base color, material properties, lighting, and normal.
pub fn calculate_lighting(
    interpolated_point: &TrianglePoint,
    view_config: &ViewConfig,
    material: &MeshMaterial, // Material properties of the surface
    mesh_mode: &MeshMode,    // Current rendering mode
) -> Color {
    let global_ambient: Color = Color::from_srgb(0.6, 0.6, 0.6);
    let global_diffuse: Color = Color::from_srgb(0.8, 0.8, 0.8);
    let global_specular: Color = Color::from_srgb(0.0, 0.0, 0.0);
    // Extract data from interpolated attributes
    let base_color_linear = interpolated_point.base_color;
    let normal_ws = interpolated_point.normal_ws;
    let vertex_position_ws = interpolated_point.position_ws;
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
    let transformation_matrix = &view_config.transformation_matrix;
    let normal_vs = normal_ws.to_view_space_as_vector(transformation_matrix);
    let vertex_position_vs = vertex_position_ws.to_view_space_as_point(transformation_matrix);
    // Normalize the normal in view space
    let mut normal_n_vs = normal_vs;
    normal_n_vs.normalize();
    // Determine light direction and light color in view-space
    let light_color = Color::white();
    let light_dir_vs = calculate_light_direction(view_config, normal_n_vs);
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
            // Ambient term now takes the surface base colour (texel) into account so that
            // ambient light is tinted appropriately.
            let ambient_comp = base_color_linear * linear_ka * global_ambient;
            // Diffuse component
            let dot_nl = normal_n_vs.dot(&light_dir_vs).max(0.0);
            let diffuse_comp = base_color_linear * linear_kd * light_color * dot_nl;
            // Ambient + Diffuse
            ambient_comp + diffuse_comp
        }
        _ => {
            // Ambient term includes the base colour (texel) so that fully lit areas retain
            // their texture hue even when only ambient illumination is present.
            let ambient_comp = base_color_linear * linear_ka * global_ambient;
            // Diffuse component
            let dot_nl = normal_n_vs.dot(&light_dir_vs).max(0.0);
            let diffuse_comp =
                base_color_linear * linear_kd * light_color * dot_nl * global_diffuse;
            // Specular component (Blinn-Phong)
            let mut halfway_dir = light_dir_vs + view_dir_vs;
            halfway_dir.normalize();
            let dot_nh = normal_n_vs.dot(&halfway_dir).max(0.0);
            let specular_intensity = if material.ns > 0.0 {
                dot_nh.powf(material.ns)
            } else {
                0.0
            };
            let specular_comp = linear_ks * light_color * specular_intensity * global_specular;
            // Full model: Ambient + Diffuse + Specular (illum >= 2)
            ambient_comp + diffuse_comp + specular_comp
        }
    };
    // Clamp to valid range and return the final color
    let final_color_clamped = final_color_linear.clamp(0.0, 1.0);
    final_color_clamped
}
