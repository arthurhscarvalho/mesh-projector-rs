//! This module provides functions for physically-based lighting calculations,
//! including support for different light sources, material properties, and
//! color space conversions. It is used to compute the final color of surfaces
//! in 3D scenes, taking into account ambient, diffuse, and specular lighting
//! components, as well as transformations between world and view space.

use crate::projector::config::LightSource;
use crate::projector::constants::GLOBAL_AMBIENT;
use crate::projector::math::{subtract_vec3, transform_point_3d, transform_vector_3d};
use crate::types::color::Color;
use crate::types::coordinate::Coordinate;
use crate::types::mesh::{MeshMaterial, MeshMode};
use ndarray::Array2;

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
    let linear_ka = Color::from(material.ka).to_linear();
    let linear_kd = Color::from(material.kd).to_linear();
    let linear_ks = Color::from(material.ks).to_linear();

    // Transform normal and vertex position to view space
    let normal_vs = transform_vector_3d(Coordinate::from(normal_ws), view_matrix); // Normals are vectors (w=0)
    let vertex_position_vs = transform_point_3d(Coordinate::from(vertex_position_ws), view_matrix); // Points are transformed (w=1)

    // Normalize the normal in view space
    let mut normal_n_vs = normal_vs;
    normal_n_vs.normalize();

    // Determine light direction and light color in view-space
    let (light_dir_vs, light_color) = match light_source {
        LightSource::World(world_light_pos) => {
            // Treat world_light_pos as a direction vector in world space and transform it
            let mut dir_vs = transform_vector_3d(*world_light_pos, view_matrix);
            dir_vs.normalize();
            (dir_vs, Color::white()) // White light
        }
        LightSource::Camera => {
            // For orthographic projection with camera lighting, the light direction should
            // match the view direction in world space (transformed to view space)
            // This ensures light comes from the projection plane for any view angle
            let mut dir_ws = Coordinate::from(view_dir_ws); // Use the passed view direction
            dir_ws.normalize();

            // Transform the world-space view direction to view space
            let mut dir_vs = transform_vector_3d(dir_ws, view_matrix);
            dir_vs.normalize();

            // Negate to make light point toward surface from camera
            dir_vs = -dir_vs;

            (dir_vs, Color::white()) // White light
        }
    };

    // View direction in view space is from surface point to camera (origin in VS)
    let mut view_dir_vs = subtract_vec3(Coordinate::new(0.0, 0.0, 0.0), vertex_position_vs);
    view_dir_vs.normalize();

    let final_color_linear = match material.illum {
        0 => {
            // Constant color - ambient only, modulated by base color
            Color::from(base_color_linear) * Color::new(linear_ka.r(), linear_ka.g(), linear_ka.b())
        }
        1 => {
            // Ambient + Diffuse

            // Ambient component using global ambient and material ambient
            let ambient_comp = GLOBAL_AMBIENT * linear_ka;

            // Diffuse component
            let dot_nl = normal_n_vs.dot(&light_dir_vs).max(0.0);
            let diffuse_comp = Color::from(base_color_linear) * linear_kd * dot_nl * light_color;

            ambient_comp + diffuse_comp
        }
        _ => {
            // Full model: Ambient + Diffuse + Specular (illum >= 2)

            // Ambient component using global ambient and material ambient
            let ambient_comp = GLOBAL_AMBIENT * linear_ka;

            // Diffuse component
            let dot_nl = normal_n_vs.dot(&light_dir_vs).max(0.0);
            let diffuse_comp = Color::from(base_color_linear) * linear_kd * dot_nl * light_color;

            // Specular component (Blinn-Phong)
            let mut halfway_dir = light_dir_vs + view_dir_vs;
            halfway_dir.normalize();
            let dot_nh = normal_n_vs.dot(&halfway_dir).max(0.0);

            let specular_intensity = if material.ns > 0.0 {
                dot_nh.powf(material.ns)
            } else {
                0.0
            };

            let specular_comp = linear_ks * specular_intensity * light_color;

            ambient_comp + diffuse_comp + specular_comp
        }
    };

    // Clamp to valid range and convert back to array
    let final_color_clamped = final_color_linear.clamp(0.0, 1.0);
    final_color_clamped.into()
}

// Removed placeholder transform_normal_ws_to_vs as we use transform_vector_3d
