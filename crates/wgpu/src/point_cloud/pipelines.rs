use crate::gpu::bindings::{storage_entry, uniform_entry};
use crate::gpu::shader;
use std::borrow::Cow;

pub(crate) struct PointCloudPipelineResources {
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pub(crate) pipeline: wgpu::ComputePipeline,
}

pub(crate) fn create_point_cloud_pipeline_resources(
    device: &wgpu::Device,
) -> PointCloudPipelineResources {
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

    PointCloudPipelineResources {
        bind_group_layout,
        pipeline,
    }
}
