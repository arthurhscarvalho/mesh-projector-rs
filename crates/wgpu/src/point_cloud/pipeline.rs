//! Point-cloud compute pipeline.

use crate::error::{WgpuProjectorError, WgpuResult};
use crate::gpu::bindings::{bind_entry, storage_entry, uniform_entry};
use crate::gpu::buffers;
use crate::gpu::formats::POINT_CLOUD_WORKGROUP_SIZE;
use crate::gpu::shader;
use crate::point_cloud::input::{build_params, prepare_point_cloud_input};
use mesh_projector_core::config::{LightingConfig, PointCloudConfig};
use mesh_projector_core::types::mesh::Scene;
use ndarray::Array2;
use std::borrow::Cow;
use wgpu::util::DeviceExt;

pub(crate) fn project_to_point_cloud_by_sampling(
    instance: &wgpu::Instance,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mesh: &Scene,
    settings: &PointCloudConfig,
    lighting_config: &LightingConfig,
) -> WgpuResult<(Array2<f32>, Array2<u8>)> {
    let input = prepare_point_cloud_input(mesh, settings)?;
    if input.total_points == 0 {
        return Ok((Array2::zeros((0, 3)), Array2::zeros((0, 3))));
    }

    let total_points_u32 =
        u32::try_from(input.total_points).map_err(|_| WgpuProjectorError::OutputTooLarge {
            points: input.total_points,
        })?;
    let params = build_params(
        lighting_config,
        input.triangles.len() as u32,
        total_points_u32,
    )?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh-projector point cloud shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader::POINT_CLOUD)),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-projector point cloud bind group layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, true),
            storage_entry(3, true),
            storage_entry(4, false),
            storage_entry(5, false),
            uniform_entry(6),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mesh-projector point cloud pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mesh-projector point cloud pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let triangle_buffer = buffers::create_storage_buffer(
        device,
        "mesh-projector point cloud triangles",
        bytemuck::cast_slice(&input.triangles),
    );
    let meta_buffer = buffers::create_storage_buffer(
        device,
        "mesh-projector point cloud triangle metadata",
        bytemuck::cast_slice(&input.metas),
    );
    let texel_buffer = buffers::create_storage_buffer(
        device,
        "mesh-projector point cloud texels",
        bytemuck::cast_slice(&input.texels),
    );
    let offset_buffer = buffers::create_storage_buffer(
        device,
        "mesh-projector point cloud offsets",
        bytemuck::cast_slice(&input.offsets),
    );
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh-projector point cloud params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let positions_size =
        (input.total_points * std::mem::size_of::<[f32; 4]>()) as wgpu::BufferAddress;
    let colors_size = (input.total_points * std::mem::size_of::<u32>()) as wgpu::BufferAddress;
    let positions_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mesh-projector point cloud output positions"),
        size: positions_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let colors_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mesh-projector point cloud output colors"),
        size: colors_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let positions_readback = buffers::create_readback_buffer(
        device,
        "mesh-projector point cloud positions readback",
        positions_size,
    );
    let colors_readback = buffers::create_readback_buffer(
        device,
        "mesh-projector point cloud colors readback",
        colors_size,
    );

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh-projector point cloud bind group"),
        layout: &bind_group_layout,
        entries: &[
            bind_entry(0, &triangle_buffer),
            bind_entry(1, &meta_buffer),
            bind_entry(2, &texel_buffer),
            bind_entry(3, &offset_buffer),
            bind_entry(4, &positions_buffer),
            bind_entry(5, &colors_buffer),
            bind_entry(6, &params_buffer),
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mesh-projector point cloud encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mesh-projector point cloud compute pass"),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = total_points_u32.div_ceil(POINT_CLOUD_WORKGROUP_SIZE);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&positions_buffer, 0, &positions_readback, 0, positions_size);
    encoder.copy_buffer_to_buffer(&colors_buffer, 0, &colors_readback, 0, colors_size);
    queue.submit(Some(encoder.finish()));

    let positions_bytes = buffers::read_buffer(instance, &positions_readback)?;
    let colors_bytes = buffers::read_buffer(instance, &colors_readback)?;

    let positions_padded: &[[f32; 4]] = bytemuck::cast_slice(&positions_bytes);
    let colors_packed: &[u32] = bytemuck::cast_slice(&colors_bytes);
    let mut points = Vec::with_capacity(input.total_points * 3);
    let mut colors = Vec::with_capacity(input.total_points * 3);
    for point in positions_padded.iter().take(input.total_points) {
        points.extend_from_slice(&point[0..3]);
    }
    for packed in colors_packed.iter().take(input.total_points) {
        colors.push((packed & 255) as u8);
        colors.push(((packed >> 8) & 255) as u8);
        colors.push(((packed >> 16) & 255) as u8);
    }

    Ok((
        Array2::from_shape_vec((input.total_points, 3), points)?,
        Array2::from_shape_vec((input.total_points, 3), colors)?,
    ))
}
