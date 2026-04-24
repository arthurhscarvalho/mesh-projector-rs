use crate::gpu::bindings::{
    color_target, multisampled_texture_entry, storage_entry_for, storage_texture_entry,
    texture_entry, uniform_entry_for,
};
use crate::gpu::formats::{DEPTH_FORMAT, R8_FORMAT, RAW_DEPTH_FORMAT, RGBA_FORMAT};
use crate::gpu::shader;
use crate::image::output::WgpuOutputSelection;
use crate::image::prepared_scene::GpuImageVertex;
use std::borrow::Cow;

pub(crate) struct ImagePipelineResources {
    pub(crate) bind_group_layout: wgpu::BindGroupLayout,
    pipelines: Vec<ImageRenderPipelineEntry>,
    pub(crate) resolve: ResolvePipelineResources,
    pub(crate) ssao: SsaoPipelineResources,
    pub(crate) pack: PackPipelineResources,
}

struct ImageRenderPipelineEntry {
    selection: WgpuOutputSelection,
    sample_count: u32,
    pipeline: wgpu::RenderPipeline,
}

pub(crate) struct SsaoPipelineResources {
    pub(crate) occlusion_layout: wgpu::BindGroupLayout,
    pub(crate) blur_layout: wgpu::BindGroupLayout,
    pub(crate) darken_layout: wgpu::BindGroupLayout,
    pub(crate) occlusion_pipeline: wgpu::ComputePipeline,
    pub(crate) blur_pipeline: wgpu::ComputePipeline,
    pub(crate) darken_pipeline: wgpu::ComputePipeline,
}

pub(crate) struct ResolvePipelineResources {
    pub(crate) gbuffer_layout: wgpu::BindGroupLayout,
    pub(crate) gbuffer_pipeline: wgpu::ComputePipeline,
    pub(crate) raw_depth_layout: wgpu::BindGroupLayout,
    pub(crate) raw_depth_pipeline: wgpu::ComputePipeline,
}

pub(crate) struct PackPipelineResources {
    pub(crate) layout: wgpu::BindGroupLayout,
    pub(crate) pipeline: wgpu::ComputePipeline,
}

impl ImagePipelineResources {
    pub(crate) fn pipeline_for(
        &self,
        selection: WgpuOutputSelection,
        sample_count: u32,
    ) -> &wgpu::RenderPipeline {
        self.pipelines
            .iter()
            .find(|entry| entry.selection == selection && entry.sample_count == sample_count)
            .map(|entry| &entry.pipeline)
            .expect("all non-empty output selections have cached pipelines")
    }
}

pub(crate) fn create_image_pipeline_resources(
    device: &wgpu::Device,
    supported_msaa_sample_counts: &[u32],
) -> ImagePipelineResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh-projector image shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader::IMAGE_PROJECT)),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-projector image bind group layout"),
        entries: &[
            storage_entry_for(wgpu::ShaderStages::FRAGMENT, 0, true),
            uniform_entry_for(wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT, 1),
            storage_entry_for(wgpu::ShaderStages::FRAGMENT, 2, true),
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mesh-projector image pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    let mut pipelines = Vec::new();
    for &sample_count in supported_msaa_sample_counts {
        for selection in all_output_selections() {
            let pipeline = create_image_render_pipeline(
                device,
                &shader,
                &pipeline_layout,
                selection,
                sample_count,
            );
            pipelines.push(ImageRenderPipelineEntry {
                selection,
                sample_count,
                pipeline,
            });
        }
    }
    let resolve = create_resolve_pipeline_resources(device);
    let ssao = create_ssao_pipeline_resources(device);
    let pack = create_pack_pipeline_resources(device);

    ImagePipelineResources {
        bind_group_layout,
        pipelines,
        resolve,
        ssao,
        pack,
    }
}

fn create_image_render_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    pipeline_layout: &wgpu::PipelineLayout,
    selection: WgpuOutputSelection,
    sample_count: u32,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mesh-projector image pipeline"),
        layout: Some(pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[GpuImageVertex::layout()],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: sample_count,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[
                Some(color_target(RGBA_FORMAT, selection.color)),
                Some(color_target(R8_FORMAT, selection.depth)),
                Some(color_target(RGBA_FORMAT, selection.normal)),
                Some(color_target(R8_FORMAT, selection.occupancy)),
                Some(color_target(RAW_DEPTH_FORMAT, true)),
            ],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_resolve_pipeline_resources(device: &wgpu::Device) -> ResolvePipelineResources {
    let gbuffer_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh-projector MSAA resolve shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader::IMAGE_MSAA_RESOLVE)),
    });
    let raw_depth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh-projector MSAA raw-depth resolve shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader::IMAGE_MSAA_RAW_DEPTH_RESOLVE)),
    });
    let gbuffer_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-projector MSAA G-buffer resolve bind group layout"),
        entries: &[
            multisampled_texture_entry(0, RGBA_FORMAT),
            multisampled_texture_entry(1, R8_FORMAT),
            multisampled_texture_entry(2, RGBA_FORMAT),
            multisampled_texture_entry(3, R8_FORMAT),
            multisampled_texture_entry(4, RAW_DEPTH_FORMAT),
            storage_texture_entry(5, RGBA_FORMAT),
            storage_texture_entry(6, RGBA_FORMAT),
            storage_texture_entry(7, RGBA_FORMAT),
            storage_texture_entry(8, RGBA_FORMAT),
            uniform_entry_for(wgpu::ShaderStages::COMPUTE, 9),
        ],
    });
    let gbuffer_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mesh-projector MSAA G-buffer resolve pipeline layout"),
        bind_group_layouts: &[Some(&gbuffer_layout)],
        immediate_size: 0,
    });
    let gbuffer_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mesh-projector MSAA G-buffer resolve pipeline"),
        layout: Some(&gbuffer_pipeline_layout),
        module: &gbuffer_shader,
        entry_point: Some("resolve_msaa_gbuffer"),
        compilation_options: Default::default(),
        cache: None,
    });

    let raw_depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-projector MSAA raw-depth resolve bind group layout"),
        entries: &[
            multisampled_texture_entry(0, RAW_DEPTH_FORMAT),
            multisampled_texture_entry(1, R8_FORMAT),
            storage_texture_entry(2, RAW_DEPTH_FORMAT),
            uniform_entry_for(wgpu::ShaderStages::COMPUTE, 3),
        ],
    });
    let raw_depth_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh-projector MSAA raw-depth resolve pipeline layout"),
            bind_group_layouts: &[Some(&raw_depth_layout)],
            immediate_size: 0,
        });
    let raw_depth_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mesh-projector MSAA raw-depth resolve pipeline"),
        layout: Some(&raw_depth_pipeline_layout),
        module: &raw_depth_shader,
        entry_point: Some("resolve_raw_depth"),
        compilation_options: Default::default(),
        cache: None,
    });

    ResolvePipelineResources {
        gbuffer_layout,
        gbuffer_pipeline,
        raw_depth_layout,
        raw_depth_pipeline,
    }
}

fn create_pack_pipeline_resources(device: &wgpu::Device) -> PackPipelineResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh-projector pack shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader::IMAGE_PACK)),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-projector pack bind group layout"),
        entries: &[
            texture_entry(0, RGBA_FORMAT),
            texture_entry(1, RGBA_FORMAT),
            texture_entry(2, RGBA_FORMAT),
            texture_entry(3, RGBA_FORMAT),
            storage_entry_for(wgpu::ShaderStages::COMPUTE, 4, false),
            storage_entry_for(wgpu::ShaderStages::COMPUTE, 5, false),
            storage_entry_for(wgpu::ShaderStages::COMPUTE, 6, false),
            storage_entry_for(wgpu::ShaderStages::COMPUTE, 7, false),
            uniform_entry_for(wgpu::ShaderStages::COMPUTE, 8),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mesh-projector pack pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mesh-projector pack pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("pack_outputs"),
        compilation_options: Default::default(),
        cache: None,
    });
    PackPipelineResources { layout, pipeline }
}

fn create_ssao_pipeline_resources(device: &wgpu::Device) -> SsaoPipelineResources {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh-projector ssao shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader::IMAGE_SSAO)),
    });
    let empty_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-projector empty bind group layout"),
        entries: &[],
    });
    let occlusion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-projector ssao occlusion bind group layout"),
        entries: &[
            texture_entry(0, RAW_DEPTH_FORMAT),
            texture_entry(1, RGBA_FORMAT),
            texture_entry(2, RGBA_FORMAT),
            storage_texture_entry(3, RAW_DEPTH_FORMAT),
            uniform_entry_for(wgpu::ShaderStages::COMPUTE, 4),
        ],
    });
    let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-projector ssao blur bind group layout"),
        entries: &[
            texture_entry(0, RAW_DEPTH_FORMAT),
            storage_texture_entry(1, RAW_DEPTH_FORMAT),
            uniform_entry_for(wgpu::ShaderStages::COMPUTE, 2),
        ],
    });
    let darken_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mesh-projector ssao darken bind group layout"),
        entries: &[
            texture_entry(0, RGBA_FORMAT),
            texture_entry(1, RGBA_FORMAT),
            texture_entry(2, RAW_DEPTH_FORMAT),
            storage_texture_entry(3, RGBA_FORMAT),
            uniform_entry_for(wgpu::ShaderStages::COMPUTE, 4),
        ],
    });

    let occlusion_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh-projector ssao occlusion pipeline layout"),
            bind_group_layouts: &[Some(&occlusion_layout)],
            immediate_size: 0,
        });
    let blur_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mesh-projector ssao blur pipeline layout"),
        bind_group_layouts: &[Some(&empty_layout), Some(&blur_layout)],
        immediate_size: 0,
    });
    let darken_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mesh-projector ssao darken pipeline layout"),
        bind_group_layouts: &[
            Some(&empty_layout),
            Some(&empty_layout),
            Some(&darken_layout),
        ],
        immediate_size: 0,
    });

    let occlusion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mesh-projector ssao occlusion pipeline"),
        layout: Some(&occlusion_pipeline_layout),
        module: &shader,
        entry_point: Some("compute_occlusion"),
        compilation_options: Default::default(),
        cache: None,
    });
    let blur_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mesh-projector ssao blur pipeline"),
        layout: Some(&blur_pipeline_layout),
        module: &shader,
        entry_point: Some("blur_occlusion"),
        compilation_options: Default::default(),
        cache: None,
    });
    let darken_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("mesh-projector ssao darken pipeline"),
        layout: Some(&darken_pipeline_layout),
        module: &shader,
        entry_point: Some("darken_color"),
        compilation_options: Default::default(),
        cache: None,
    });

    SsaoPipelineResources {
        occlusion_layout,
        blur_layout,
        darken_layout,
        occlusion_pipeline,
        blur_pipeline,
        darken_pipeline,
    }
}

fn all_output_selections() -> [WgpuOutputSelection; 15] {
    [
        WgpuOutputSelection {
            color: true,
            depth: false,
            normal: false,
            occupancy: false,
        },
        WgpuOutputSelection {
            color: false,
            depth: true,
            normal: false,
            occupancy: false,
        },
        WgpuOutputSelection {
            color: false,
            depth: false,
            normal: true,
            occupancy: false,
        },
        WgpuOutputSelection {
            color: false,
            depth: false,
            normal: false,
            occupancy: true,
        },
        WgpuOutputSelection {
            color: true,
            depth: true,
            normal: false,
            occupancy: false,
        },
        WgpuOutputSelection {
            color: true,
            depth: false,
            normal: true,
            occupancy: false,
        },
        WgpuOutputSelection {
            color: true,
            depth: false,
            normal: false,
            occupancy: true,
        },
        WgpuOutputSelection {
            color: false,
            depth: true,
            normal: true,
            occupancy: false,
        },
        WgpuOutputSelection {
            color: false,
            depth: true,
            normal: false,
            occupancy: true,
        },
        WgpuOutputSelection {
            color: false,
            depth: false,
            normal: true,
            occupancy: true,
        },
        WgpuOutputSelection {
            color: true,
            depth: true,
            normal: true,
            occupancy: false,
        },
        WgpuOutputSelection {
            color: true,
            depth: true,
            normal: false,
            occupancy: true,
        },
        WgpuOutputSelection {
            color: true,
            depth: false,
            normal: true,
            occupancy: true,
        },
        WgpuOutputSelection {
            color: false,
            depth: true,
            normal: true,
            occupancy: true,
        },
        WgpuOutputSelection::all(),
    ]
}
