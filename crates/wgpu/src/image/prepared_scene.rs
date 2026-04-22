use crate::error::WgpuResult;
use crate::gpu::constants::{shading_mode_id, vec4};
use bytemuck::{Pod, Zeroable};
use mesh_projector_core::geometry::extract_triangle_surface;
use mesh_projector_core::types::coordinate::Coordinate;
use mesh_projector_core::types::mesh::{Scene, ShadingMode};
use std::mem;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuImageVertex {
    pub(crate) position: [f32; 4],
    pub(crate) normal: [f32; 4],
    pub(crate) uv_color_rg: [f32; 4],
    pub(crate) color_b_material: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuImageMaterial {
    pub(crate) ka: [f32; 4],
    pub(crate) kd: [f32; 4],
    pub(crate) ks_ns: [f32; 4],
    pub(crate) tex: [f32; 4],
    pub(crate) mode_illum_opacity: [f32; 4],
}

impl GpuImageVertex {
    pub(crate) const ATTRIBUTES: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x4,
        1 => Float32x4,
        2 => Float32x4,
        3 => Float32x4
    ];

    pub(crate) fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<GpuImageVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct PreparedTriangle {
    pub(crate) first_corner: u32,
    pub(crate) face_normal: Coordinate,
    pub(crate) centroid: Coordinate,
}

pub(crate) struct ImageSceneGpuInput {
    pub(crate) vertices: Vec<GpuImageVertex>,
    pub(crate) materials: Vec<GpuImageMaterial>,
    pub(crate) triangles: Vec<PreparedTriangle>,
    pub(crate) texels: Vec<u32>,
}

/// GPU-resident static scene data for repeated image projections.
pub struct WgpuPreparedScene {
    pub(crate) projector_id: u64,
    pub(crate) vertex_buffer: wgpu::Buffer,
    pub(crate) material_buffer: wgpu::Buffer,
    pub(crate) texel_buffer: wgpu::Buffer,
    pub(crate) triangles: Vec<PreparedTriangle>,
    pub(crate) shading_mode: ShadingMode,
    pub(crate) corner_count: u32,
}

pub(crate) fn prepare_image_scene_input(mesh: &Scene) -> WgpuResult<ImageSceneGpuInput> {
    let (texels, texture_ranges) = pack_scene_textures(mesh);
    let mut vertices = Vec::new();
    let mut materials = Vec::new();
    let mut triangles = Vec::new();
    for (submesh_index, submesh) in mesh.meshes.iter().enumerate() {
        let num_faces = submesh.geometry.triangles.nrows();
        for triangle_index in 0..num_faces {
            let Ok(surface) = extract_triangle_surface(mesh, submesh_index, triangle_index) else {
                continue;
            };
            let first_corner = vertices.len() as u32;
            let (tex_offset, tex_width, tex_height, tex_channels) = texture_ranges[submesh_index];
            let material_index = materials.len() as f32;
            materials.push(GpuImageMaterial {
                ka: vec4(submesh.material.ka),
                kd: vec4(submesh.material.kd),
                ks_ns: [
                    submesh.material.ks[0],
                    submesh.material.ks[1],
                    submesh.material.ks[2],
                    submesh.material.ns,
                ],
                tex: [
                    tex_offset as f32,
                    tex_width as f32,
                    tex_height as f32,
                    tex_channels as f32,
                ],
                mode_illum_opacity: [
                    shading_mode_id(mesh.mode) as f32,
                    submesh.material.illum.max(0) as f32,
                    submesh.material.opacity.clamp(0.0, 1.0),
                    0.0,
                ],
            });
            for vertex_index in 0..3 {
                vertices.push(GpuImageVertex {
                    position: vec4(surface.vertices_ws[vertex_index].0),
                    normal: vec4(surface.normals_ws[vertex_index].0),
                    uv_color_rg: [
                        surface.uvs[vertex_index].0,
                        surface.uvs[vertex_index].1,
                        surface.colors[vertex_index].r(),
                        surface.colors[vertex_index].g(),
                    ],
                    color_b_material: [surface.colors[vertex_index].b(), material_index, 0.0, 0.0],
                });
            }
            let centroid =
                (surface.vertices_ws[0] + surface.vertices_ws[1] + surface.vertices_ws[2])
                    * (1.0 / 3.0);
            triangles.push(PreparedTriangle {
                first_corner,
                face_normal: surface.face_normal_ws,
                centroid,
            });
        }
    }
    Ok(ImageSceneGpuInput {
        vertices,
        materials,
        triangles,
        texels,
    })
}

fn pack_scene_textures(mesh: &Scene) -> (Vec<u32>, Vec<(u32, u32, u32, u32)>) {
    let mut texels = Vec::new();
    let mut texture_ranges = Vec::with_capacity(mesh.meshes.len());
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

        let offset = texels.len() as u32;
        for y in 0..shape[0] {
            for x in 0..shape[1] {
                let r = image[[y, x, 0]] as u32;
                let g = image[[y, x, 1]] as u32;
                let b = image[[y, x, 2]] as u32;
                texels.push(r | (g << 8) | (b << 16));
            }
        }
        texture_ranges.push((offset, shape[1] as u32, shape[0] as u32, shape[2] as u32));
    }

    if texels.is_empty() {
        texels.push(0);
    }

    (texels, texture_ranges)
}
