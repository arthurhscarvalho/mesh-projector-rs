//! GPU SSAO pass encoding.

use crate::gpu::bindings::{bind_entry, storage_texture_bind_entry, texture_bind_entry};
use crate::gpu::formats::{IMAGE_WORKGROUP_SIZE, RAW_DEPTH_FORMAT, RGBA_FORMAT};
use crate::image::pipelines::SsaoPipelineResources;
use crate::image::resources::{create_render_texture, RenderTextures};
use bytemuck::{Pod, Zeroable};
use mesh_projector_core::config::SsaoConfig;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSsaoParams {
    resolution: u32,
    num_samples: u32,
    blur_radius: u32,
    projection_kind: u32,
    sample_radius: f32,
    bias: f32,
    intensity: f32,
    max_distance: f32,
    focal_length: f32,
    cube_center: f32,
    _pad0: [f32; 2],
}

pub(crate) fn encode_ssao(
    device: &wgpu::Device,
    pipeline_resources: &SsaoPipelineResources,
    encoder: &mut wgpu::CommandEncoder,
    textures: &RenderTextures,
    resolution: u16,
    config: &SsaoConfig,
    projection_kind: u32,
    focal_length: f32,
    cube_center: f32,
) -> wgpu::Texture {
    let occlusion = create_render_texture(
        device,
        "mesh-projector ssao occlusion",
        resolution,
        RAW_DEPTH_FORMAT,
        1,
    );
    let blurred = create_render_texture(
        device,
        "mesh-projector ssao blurred occlusion",
        resolution,
        RAW_DEPTH_FORMAT,
        1,
    );
    let output = create_render_texture(
        device,
        "mesh-projector ssao color",
        resolution,
        RGBA_FORMAT,
        1,
    );
    let params = GpuSsaoParams {
        resolution: resolution as u32,
        num_samples: config.num_samples as u32,
        blur_radius: config.blur_radius as u32,
        projection_kind,
        sample_radius: config.sample_radius,
        bias: config.bias,
        intensity: config.intensity,
        max_distance: config.max_distance,
        focal_length,
        cube_center,
        _pad0: [0.0; 2],
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh-projector ssao params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let raw_depth_view = textures.raw_depth.create_view(&Default::default());
    let normal_view = textures.normal.create_view(&Default::default());
    let occupancy_view = textures.occupancy.create_view(&Default::default());
    let occlusion_view = occlusion.create_view(&Default::default());
    let blurred_view = blurred.create_view(&Default::default());
    let color_view = textures.color.create_view(&Default::default());
    let output_view = output.create_view(&Default::default());

    let occlusion_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh-projector ssao occlusion bind group"),
        layout: &pipeline_resources.occlusion_layout,
        entries: &[
            texture_bind_entry(0, &raw_depth_view),
            texture_bind_entry(1, &normal_view),
            texture_bind_entry(2, &occupancy_view),
            storage_texture_bind_entry(3, &occlusion_view),
            bind_entry(4, &params_buffer),
        ],
    });
    let blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh-projector ssao blur bind group"),
        layout: &pipeline_resources.blur_layout,
        entries: &[
            texture_bind_entry(0, &occlusion_view),
            storage_texture_bind_entry(1, &blurred_view),
            bind_entry(2, &params_buffer),
        ],
    });
    let darken_source = if config.blur_radius > 0 {
        &blurred_view
    } else {
        &occlusion_view
    };
    let darken_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh-projector ssao darken bind group"),
        layout: &pipeline_resources.darken_layout,
        entries: &[
            texture_bind_entry(0, &color_view),
            texture_bind_entry(1, &occupancy_view),
            texture_bind_entry(2, darken_source),
            storage_texture_bind_entry(3, &output_view),
            bind_entry(4, &params_buffer),
        ],
    });
    let workgroups = (resolution as u32).div_ceil(IMAGE_WORKGROUP_SIZE);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mesh-projector ssao compute pass"),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline_resources.occlusion_pipeline);
        pass.set_bind_group(0, &occlusion_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
        if config.blur_radius > 0 {
            pass.set_pipeline(&pipeline_resources.blur_pipeline);
            pass.set_bind_group(1, &blur_bind_group, &[]);
            pass.dispatch_workgroups(workgroups, workgroups, 1);
        }
        pass.set_pipeline(&pipeline_resources.darken_pipeline);
        pass.set_bind_group(2, &darken_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    output
}
