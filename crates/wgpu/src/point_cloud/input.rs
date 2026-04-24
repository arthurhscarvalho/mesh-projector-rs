//! Point-cloud GPU input preparation.

use crate::error::{WgpuProjectorError, WgpuResult};
use crate::gpu::constants::{
    shading_mode_id, vec4, LIGHT_AMBIENT, LIGHT_DIRECTIONAL, LIGHT_HEADLIGHT, LIGHT_POINT,
    MAX_POINT_LIGHTS,
};
use bytemuck::{Pod, Zeroable};
use mesh_projector_core::config::{LightSource, LightingConfig, PointCloudConfig};
use mesh_projector_core::geometry::{calculate_triangle_area, extract_triangle_surface};
use mesh_projector_core::types::mesh::Scene;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuTriangle {
    pub(crate) v0: [f32; 4],
    pub(crate) v1: [f32; 4],
    pub(crate) v2: [f32; 4],
    pub(crate) n0: [f32; 4],
    pub(crate) n1: [f32; 4],
    pub(crate) n2: [f32; 4],
    pub(crate) uv0: [f32; 4],
    pub(crate) uv1: [f32; 4],
    pub(crate) uv2: [f32; 4],
    pub(crate) c0: [f32; 4],
    pub(crate) c1: [f32; 4],
    pub(crate) c2: [f32; 4],
    pub(crate) ka: [f32; 4],
    pub(crate) kd: [f32; 4],
    pub(crate) ks_ns: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuTriangleMeta {
    pub(crate) tex_offset: u32,
    pub(crate) tex_width: u32,
    pub(crate) tex_height: u32,
    pub(crate) tex_channels: u32,
    pub(crate) n: u32,
    pub(crate) shading_mode: u32,
    pub(crate) illum: u32,
    pub(crate) _pad0: u32,
}

pub(crate) struct PointCloudGpuInput {
    pub(crate) triangles: Vec<GpuTriangle>,
    pub(crate) metas: Vec<GpuTriangleMeta>,
    pub(crate) offsets: Vec<u32>,
    pub(crate) texels: Vec<u32>,
    pub(crate) total_points: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuParams {
    pub(crate) global_ambient: [f32; 4],
    pub(crate) global_diffuse: [f32; 4],
    pub(crate) global_specular: [f32; 4],
    pub(crate) directional_light: [f32; 4],
    pub(crate) view_direction: [f32; 4],
    pub(crate) point_lights: [[f32; 4]; MAX_POINT_LIGHTS],
    pub(crate) light_kind: u32,
    pub(crate) light_count: u32,
    pub(crate) triangle_count: u32,
    pub(crate) total_points: u32,
}

pub(crate) fn prepare_point_cloud_input(
    mesh: &Scene,
    settings: &PointCloudConfig,
) -> WgpuResult<PointCloudGpuInput> {
    let mut triangles = Vec::new();
    let mut metas = Vec::new();
    let mut offsets = vec![0u32];
    let mut texels = Vec::new();
    let mut total_points = 0usize;

    for submesh in &mesh.meshes {
        if let Some(texture) = &submesh.texture {
            if let Some(image) = &texture.texture {
                if texture.width_f32 > 0.0 && texture.height_f32 > 0.0 {
                    let shape = image.shape();
                    let channels = shape[2];
                    let base_offset = texels.len() as u32;
                    for y in 0..shape[0] {
                        for x in 0..shape[1] {
                            let r = image[[y, x, 0]] as u32;
                            let g = image[[y, x, 1]] as u32;
                            let b = image[[y, x, 2]] as u32;
                            texels.push(r | (g << 8) | (b << 16));
                        }
                    }
                    let _ = (base_offset, channels);
                }
            }
        }
    }

    if texels.is_empty() {
        texels.push(0);
    }

    let mut texture_ranges = Vec::with_capacity(mesh.meshes.len());
    let mut next_texel_offset = 0u32;
    for submesh in &mesh.meshes {
        let Some(texture) = &submesh.texture else {
            texture_ranges.push((0, 0, 0, 0));
            continue;
        };
        let Some(image) = &texture.texture else {
            texture_ranges.push((0, 0, 0, 0));
            continue;
        };
        let shape = image.shape();
        if shape[0] == 0 || shape[1] == 0 || shape[2] < 3 {
            texture_ranges.push((0, 0, 0, 0));
            continue;
        }
        let width = shape[1] as u32;
        let height = shape[0] as u32;
        let channels = shape[2] as u32;
        texture_ranges.push((next_texel_offset, width, height, channels));
        next_texel_offset += width * height;
    }

    for (submesh_index, submesh) in mesh.meshes.iter().enumerate() {
        let num_faces = submesh.geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            let Ok(surface) = extract_triangle_surface(mesh, submesh_index, triangle_index) else {
                continue;
            };
            let triangle_area = calculate_triangle_area(&surface.vertices_ws);
            let n = {
                let s = triangle_area * settings.point_cloud_density_factor;
                ((-3.0 + (1.0 + 8.0 * s).sqrt()) * 0.5).ceil() as u32
            };
            let sample_count = (((n + 1) as usize) * ((n + 2) as usize)) / 2;
            if total_points + sample_count > u32::MAX as usize {
                return Err(WgpuProjectorError::OutputTooLarge {
                    points: total_points + sample_count,
                });
            }
            total_points += sample_count;
            offsets.push(total_points as u32);

            let (tex_offset, tex_width, tex_height, tex_channels) = texture_ranges[submesh_index];
            triangles.push(GpuTriangle {
                v0: vec4(surface.vertices_ws[0].0),
                v1: vec4(surface.vertices_ws[1].0),
                v2: vec4(surface.vertices_ws[2].0),
                n0: vec4(surface.normals_ws[0].0),
                n1: vec4(surface.normals_ws[1].0),
                n2: vec4(surface.normals_ws[2].0),
                uv0: [surface.uvs[0].0, surface.uvs[0].1, 0.0, 0.0],
                uv1: [surface.uvs[1].0, surface.uvs[1].1, 0.0, 0.0],
                uv2: [surface.uvs[2].0, surface.uvs[2].1, 0.0, 0.0],
                c0: vec4(surface.colors[0].0),
                c1: vec4(surface.colors[1].0),
                c2: vec4(surface.colors[2].0),
                ka: vec4(submesh.material.ka),
                kd: vec4(submesh.material.kd),
                ks_ns: [
                    submesh.material.ks[0],
                    submesh.material.ks[1],
                    submesh.material.ks[2],
                    submesh.material.ns,
                ],
            });
            metas.push(GpuTriangleMeta {
                tex_offset,
                tex_width,
                tex_height,
                tex_channels,
                n,
                shading_mode: shading_mode_id(mesh.mode),
                illum: submesh.material.illum.max(0) as u32,
                _pad0: 0,
            });
        }
    }

    Ok(PointCloudGpuInput {
        triangles,
        metas,
        offsets,
        texels,
        total_points,
    })
}

pub(crate) fn build_params(
    lighting_config: &LightingConfig,
    triangle_count: u32,
    total_points: u32,
) -> WgpuResult<GpuParams> {
    let mut params = GpuParams {
        global_ambient: vec4(lighting_config.components.global_ambient.0),
        global_diffuse: vec4(lighting_config.components.global_diffuse.0),
        global_specular: vec4(lighting_config.components.global_specular.0),
        directional_light: [0.0, 0.0, 1.0, 0.0],
        view_direction: [0.0, 0.0, 1.0, 0.0],
        point_lights: [[0.0; 4]; MAX_POINT_LIGHTS],
        light_kind: LIGHT_AMBIENT,
        light_count: 0,
        triangle_count,
        total_points,
    };

    match &lighting_config.source {
        LightSource::Ambient => {
            params.light_kind = LIGHT_AMBIENT;
        }
        LightSource::Directional(direction) => {
            params.light_kind = LIGHT_DIRECTIONAL;
            params.directional_light = vec4(direction.0);
        }
        LightSource::Headlight => {
            params.light_kind = LIGHT_HEADLIGHT;
        }
        LightSource::Point(points) => {
            if points.len() > MAX_POINT_LIGHTS {
                return Err(WgpuProjectorError::TooManyPointLights {
                    count: points.len(),
                    max: MAX_POINT_LIGHTS,
                });
            }
            params.light_kind = LIGHT_POINT;
            params.light_count = points.len() as u32;
            for (index, point) in points.iter().enumerate() {
                params.point_lights[index] = vec4(point.0);
            }
        }
    }
    Ok(params)
}
