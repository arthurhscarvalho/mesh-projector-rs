//! Point-cloud compute pipeline.

use crate::error::{WgpuProjectorError, WgpuResult};
use crate::gpu::bindings::bind_entry;
use crate::gpu::buffers;
use crate::gpu::formats::POINT_CLOUD_WORKGROUP_SIZE;
use crate::point_cloud::input::{build_params, prepare_point_cloud_input};
use crate::point_cloud::output::WgpuOwnedMappedPointCloudReadback;
use crate::point_cloud::pipelines::PointCloudPipelineResources;
use crate::point_cloud::prepared_scene::WgpuPreparedPointCloudScene;
use crate::point_cloud::readback::{
    into_owned_mapped_point_cloud_readback, schedule_point_cloud_readback_maps,
};
use crate::point_cloud::resources::{
    create_point_cloud_output_buffers, enqueue_point_cloud_readback_copies,
    PointCloudReadbackLeaseGuard, PointCloudReadbackPoolHandle,
};
use crate::point_cloud::timings::WgpuPointCloudTimings;
use mesh_projector_core::config::{LightingConfig, PointCloudConfig};
use mesh_projector_core::types::mesh::Scene;
use ndarray::Array2;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

pub(crate) fn prepare_point_cloud_scene(
    device: &wgpu::Device,
    projector_id: u64,
    mesh: &Scene,
    settings: &PointCloudConfig,
) -> WgpuResult<WgpuPreparedPointCloudScene> {
    let input = prepare_point_cloud_input(mesh, settings)?;
    let triangle_count = input.triangles.len() as u32;

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

    Ok(WgpuPreparedPointCloudScene {
        projector_id,
        triangle_buffer,
        meta_buffer,
        texel_buffer,
        offset_buffer,
        triangle_count,
        total_points: input.total_points,
    })
}

pub(crate) fn project_prepared_to_point_cloud_with_timings(
    instance: &wgpu::Instance,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &PointCloudPipelineResources,
    readback_pool: &PointCloudReadbackPoolHandle,
    prepared: &WgpuPreparedPointCloudScene,
    lighting_config: &LightingConfig,
) -> WgpuResult<((Array2<f32>, Array2<u8>), WgpuPointCloudTimings)> {
    let (readback, mut timings) = project_prepared_to_owned_point_cloud_readback(
        instance,
        device,
        queue,
        pipeline,
        readback_pool,
        prepared,
        lighting_config,
    )?;

    let copy_started = Instant::now();
    let arrays = readback.to_arrays();
    let copy_elapsed = copy_started.elapsed();
    timings.cpu_copy += copy_elapsed;
    timings.total += copy_elapsed;

    Ok((arrays, timings))
}

pub(crate) fn project_prepared_to_owned_point_cloud_readback(
    instance: &wgpu::Instance,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &PointCloudPipelineResources,
    readback_pool: &PointCloudReadbackPoolHandle,
    prepared: &WgpuPreparedPointCloudScene,
    lighting_config: &LightingConfig,
) -> WgpuResult<(WgpuOwnedMappedPointCloudReadback, WgpuPointCloudTimings)> {
    let total_started = Instant::now();
    if prepared.total_points == 0 {
        return Ok((
            WgpuOwnedMappedPointCloudReadback::zeroed(),
            WgpuPointCloudTimings {
                total: total_started.elapsed(),
                ..Default::default()
            },
        ));
    }

    let total_points_u32 =
        u32::try_from(prepared.total_points).map_err(|_| WgpuProjectorError::OutputTooLarge {
            points: prepared.total_points,
        })?;
    let params = build_params(lighting_config, prepared.triangle_count, total_points_u32)?;

    let upload_started = Instant::now();
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh-projector point cloud params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let output_buffers = create_point_cloud_output_buffers(device, prepared.total_points);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh-projector point cloud bind group"),
        layout: &pipeline.bind_group_layout,
        entries: &[
            bind_entry(0, &prepared.triangle_buffer),
            bind_entry(1, &prepared.meta_buffer),
            bind_entry(2, &prepared.texel_buffer),
            bind_entry(3, &prepared.offset_buffer),
            bind_entry(4, &output_buffers.positions),
            bind_entry(5, &output_buffers.colors),
            bind_entry(6, &params_buffer),
        ],
    });
    let upload_elapsed = upload_started.elapsed();

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mesh-projector point cloud encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mesh-projector point cloud compute pass"),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = total_points_u32.div_ceil(POINT_CLOUD_WORKGROUP_SIZE);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }

    let readback_lease = readback_pool
        .lock()
        .expect("point cloud readback pool mutex poisoned")
        .acquire(
            device,
            output_buffers.positions_size,
            output_buffers.colors_size,
        );

    let copy_to_readback_started = Instant::now();
    enqueue_point_cloud_readback_copies(&mut encoder, &output_buffers, &readback_lease);
    let scheduled = schedule_point_cloud_readback_maps(&encoder, &readback_lease);
    let copy_to_readback_elapsed = copy_to_readback_started.elapsed();

    let submit_started = Instant::now();
    queue.submit(Some(encoder.finish()));
    let compute_submit_elapsed = submit_started.elapsed();

    let readback_guard =
        PointCloudReadbackLeaseGuard::new(Arc::clone(readback_pool), readback_lease);
    let (owned_readback, readback_timings) = into_owned_mapped_point_cloud_readback(
        instance,
        &output_buffers,
        readback_guard,
        prepared.total_points,
        scheduled,
    )?;

    let timings = WgpuPointCloudTimings {
        total: total_started.elapsed(),
        prepare_scene: std::time::Duration::ZERO,
        upload: upload_elapsed,
        compute_submit: compute_submit_elapsed,
        copy_to_readback: copy_to_readback_elapsed,
        map_wait: readback_timings.map_wait,
        readback: readback_timings.map_wait,
        cpu_copy: std::time::Duration::ZERO,
        numpy_conversion: std::time::Duration::ZERO,
    };

    Ok((owned_readback, timings))
}
