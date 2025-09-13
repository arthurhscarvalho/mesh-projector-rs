//! This module provides functions for physically-based lighting calculations,
//! including support for different light sources, material properties, and
//! color space conversions. It is used to compute the final color of surfaces
//! in 3D scenes, taking into account ambient, diffuse, and specular lighting
//! components, as well as transformations between world and view space.

use crate::renderer::config::{LightSource, RenderPassConfig};
use crate::renderer::rasterization::TrianglePoint;
use crate::types::color::Color;
use crate::types::coordinate::Coordinate;
use crate::types::mesh::MeshMaterial;

fn calculate_single_light_direction(
    render_pass_config: &RenderPassConfig,
    normal_n_vs: Coordinate,
) -> Coordinate {
    let light_source = &render_pass_config.lighting.source;
    let view_dir_ws = render_pass_config.view.view_direction;
    let transformation_matrix = &render_pass_config.view.transformation_matrix;
    let light_dir_vs = match light_source {
        LightSource::Ambient => {
            // Ambient lighting: no specific light direction. We emulate uniform illumination
            // by treating the light as coming from all directions equally.
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
            let mut dir_ws = view_dir_ws;
            dir_ws.normalize();
            let mut dir_vs = dir_ws.to_view_space_as_vector(transformation_matrix);
            dir_vs.normalize();
            dir_vs = -dir_vs; // Negate to make light point toward surface from camera
            dir_vs
        }
        _ => panic!("Invalid light source, this function expects a single light source"),
    };
    light_dir_vs
}

fn flat_color_lighting(interpolated_point: &TrianglePoint) -> Color {
    let base_color_linear = interpolated_point.base_color;
    base_color_linear
}

fn lambertian_lighting(
    interpolated_point: &TrianglePoint,
    material: &MeshMaterial,
    render_pass_config: &RenderPassConfig,
) -> Color {
    let base_color_linear = interpolated_point.base_color;
    let linear_ka = Color::from(material.ka);
    let linear_kd = Color::from(material.kd);
    let global_ambient = render_pass_config.lighting.components.global_ambient;
    let normal_n_vs = interpolated_point.normal_ws;
    // Ambient component using global ambient and material ambient
    let ambient_comp = base_color_linear * linear_ka * global_ambient;
    // Diffuse component(s)
    let diffuse_total = match &render_pass_config.lighting.source {
        LightSource::Point(world_light_positions) => {
            let mut accum = Color::black();
            let transformation_matrix = &render_pass_config.view.transformation_matrix;
            for world_light_pos in world_light_positions.iter() {
                // Treat as a direction vector in world space and transform it
                let mut dir_vs = world_light_pos.to_view_space_as_vector(transformation_matrix);
                dir_vs.normalize();
                let dot_nl = normal_n_vs.dot(&dir_vs).max(0.0);
                let diffuse_comp = base_color_linear * linear_kd * dot_nl;
                accum = accum + diffuse_comp;
            }
            accum
        }
        _ => {
            let light_dir_vs = calculate_single_light_direction(render_pass_config, normal_n_vs);
            let dot_nl = normal_n_vs.dot(&light_dir_vs).max(0.0);
            base_color_linear * linear_kd * dot_nl
        }
    };
    // Ambient + Diffuse
    ambient_comp + diffuse_total
}

fn blinn_phong_lighting(
    interpolated_point: &TrianglePoint,
    render_pass_config: &RenderPassConfig,
    material: &MeshMaterial,
) -> Color {
    // Extract data from interpolated attributes
    let base_color_linear = interpolated_point.base_color;
    let normal_ws = interpolated_point.normal_ws;
    let vertex_position_ws = interpolated_point.position_ws;
    // Extract global lighting components
    let global_ambient = render_pass_config.lighting.components.global_ambient;
    let global_diffuse = render_pass_config.lighting.components.global_diffuse;
    let global_specular = render_pass_config.lighting.components.global_specular;
    // Material coefficients are stored in linear space; avoid costly per-pixel gamma conversion
    let linear_ka = Color::from(material.ka);
    let linear_kd = Color::from(material.kd);
    let linear_ks = Color::from(material.ks);
    // Transform normal and vertex position to view space
    let transformation_matrix = &render_pass_config.view.transformation_matrix;
    let normal_vs = normal_ws.to_view_space_as_vector(transformation_matrix);
    let vertex_position_vs = vertex_position_ws.to_view_space_as_point(transformation_matrix);
    // Normalize the normal in view space
    let mut normal_n_vs = normal_vs;
    normal_n_vs.normalize();
    // View direction in view space is from surface point to camera (origin in VS)
    let mut view_dir_vs = -vertex_position_vs;
    view_dir_vs.normalize();
    // Ambient component
    let ambient_comp = base_color_linear * linear_ka * global_ambient;
    // Diffuse and Specular components
    let (diffuse_total, specular_total) = match &render_pass_config.lighting.source {
        LightSource::Point(world_light_positions) => {
            let mut diffuse_accum = Color::black();
            let mut specular_accum = Color::black();
            let transformation_matrix = &render_pass_config.view.transformation_matrix;
            for world_light_pos in world_light_positions.iter() {
                // Treat as a direction vector in world space and transform it
                let mut dir_vs = world_light_pos.to_view_space_as_vector(transformation_matrix);
                dir_vs.normalize();
                // Diffuse
                let dot_nl = normal_n_vs.dot(&dir_vs).max(0.0);
                let diffuse_comp = base_color_linear * linear_kd * dot_nl * global_diffuse;
                diffuse_accum = diffuse_accum + diffuse_comp;
                // Specular (Blinn-Phong)
                let mut halfway_dir = dir_vs + view_dir_vs;
                halfway_dir.normalize();
                let dot_nh = normal_n_vs.dot(&halfway_dir).max(0.0);
                let specular_intensity = if material.ns > 0.0 {
                    dot_nh.powf(material.ns)
                } else {
                    0.0
                };
                let specular_comp = linear_ks * specular_intensity * global_specular;
                specular_accum = specular_accum + specular_comp;
            }
            (diffuse_accum, specular_accum)
        }
        _ => {
            // Determine light direction in view-space
            let light_dir_vs = calculate_single_light_direction(render_pass_config, normal_n_vs);
            let dot_nl = normal_n_vs.dot(&light_dir_vs).max(0.0);
            let diffuse_comp = base_color_linear * linear_kd * dot_nl * global_diffuse;
            // Specular (Blinn-Phong)
            let mut halfway_dir = light_dir_vs + view_dir_vs;
            halfway_dir.normalize();
            let dot_nh = normal_n_vs.dot(&halfway_dir).max(0.0);
            let specular_intensity = if material.ns > 0.0 {
                dot_nh.powf(material.ns)
            } else {
                0.0
            };
            let specular_comp = linear_ks * specular_intensity * global_specular;
            (diffuse_comp, specular_comp)
        }
    };
    // BlinnPhong model: Ambient + Diffuse + Specular (illum >= 2)
    ambient_comp + diffuse_total + specular_total
}

/// Calculates the final linear color based on base color, material properties, lighting, and normal.
pub fn calculate_lighting(
    interpolated_point: &TrianglePoint,
    render_pass_config: &RenderPassConfig,
    material: &MeshMaterial,
) -> Color {
    let final_color_linear = match material.illum {
        0 => flat_color_lighting(interpolated_point),
        1 => lambertian_lighting(interpolated_point, material, render_pass_config),
        // Default to Blinn-Phong lighting
        _ => blinn_phong_lighting(interpolated_point, render_pass_config, material),
    };
    // Clamp to valid range and return the final color
    let final_color_clamped = final_color_linear.clamp(0.0, 1.0);
    final_color_clamped
}
