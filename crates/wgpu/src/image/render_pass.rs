//! Image render, MSAA resolve, and output packing pass encoding.

use crate::gpu::bindings::{
    bind_entry, bind_entry_range, render_attachment, storage_texture_bind_entry, texture_bind_entry,
};
use crate::gpu::formats::IMAGE_WORKGROUP_SIZE;
use crate::image::output::WgpuOutputSelection;
use crate::image::pipelines::ImagePipelineResources;
use crate::image::resources::{PackedOutputGpuBuffers, RenderTextures};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuResolveParams {
    resolution: u32,
    sample_count: u32,
    _pad0: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuPackParams {
    resolution: u32,
    pixels_per_view: u32,
    view_index: u32,
    selection_bits: u32,
    dispatch_width: u32,
    color_stride_words: u32,
    scalar_words_per_view: u32,
    scalar_stride_words: u32,
}

pub(crate) fn encode_image_view(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    index_count: u32,
    background: [u8; 4],
    selection: WgpuOutputSelection,
    textures: &RenderTextures,
) {
    let color_attachment = textures
        .color_msaa
        .as_ref()
        .unwrap_or(&textures.color)
        .create_view(&Default::default());
    let depth_byte_attachment = textures
        .depth_byte_msaa
        .as_ref()
        .unwrap_or(&textures.depth_byte)
        .create_view(&Default::default());
    let normal_attachment = textures
        .normal_msaa
        .as_ref()
        .unwrap_or(&textures.normal)
        .create_view(&Default::default());
    let occupancy_attachment = textures
        .occupancy_msaa
        .as_ref()
        .unwrap_or(&textures.occupancy)
        .create_view(&Default::default());
    let raw_depth_attachment = textures
        .raw_depth_msaa
        .as_ref()
        .unwrap_or(&textures.raw_depth)
        .create_view(&Default::default());
    let depth_view = textures.depth.create_view(&Default::default());
    let clear = wgpu::Color {
        r: background[0] as f64 / 255.0,
        g: background[1] as f64 / 255.0,
        b: background[2] as f64 / 255.0,
        a: background[3] as f64 / 255.0,
    };

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("mesh-projector image render pass"),
        color_attachments: &[
            Some(render_attachment(
                &color_attachment,
                None,
                clear,
                selection.color,
            )),
            Some(render_attachment(
                &depth_byte_attachment,
                None,
                wgpu::Color::BLACK,
                selection.depth,
            )),
            Some(render_attachment(
                &normal_attachment,
                None,
                wgpu::Color::BLACK,
                selection.normal,
            )),
            Some(render_attachment(
                &occupancy_attachment,
                None,
                wgpu::Color::BLACK,
                selection.occupancy,
            )),
            Some(render_attachment(
                &raw_depth_attachment,
                None,
                wgpu::Color::BLACK,
                true,
            )),
        ],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.set_vertex_buffer(0, vertex_buffer.slice(..));
    pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..index_count, 0, 0..1);
}

pub(crate) fn encode_msaa_resolve(
    device: &wgpu::Device,
    pipeline_resources: &ImagePipelineResources,
    encoder: &mut wgpu::CommandEncoder,
    textures: &RenderTextures,
    resolution: u16,
    sample_count: u32,
) {
    let Some(color_msaa) = &textures.color_msaa else {
        return;
    };
    let Some(depth_byte_msaa) = &textures.depth_byte_msaa else {
        return;
    };
    let Some(normal_msaa) = &textures.normal_msaa else {
        return;
    };
    let Some(occupancy_msaa) = &textures.occupancy_msaa else {
        return;
    };
    let Some(raw_depth_msaa) = &textures.raw_depth_msaa else {
        return;
    };
    let params = GpuResolveParams {
        resolution: resolution as u32,
        sample_count,
        _pad0: [0; 2],
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh-projector MSAA resolve params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let color_msaa_view = color_msaa.create_view(&Default::default());
    let depth_msaa_view = depth_byte_msaa.create_view(&Default::default());
    let normal_msaa_view = normal_msaa.create_view(&Default::default());
    let occupancy_msaa_view = occupancy_msaa.create_view(&Default::default());
    let raw_depth_msaa_view = raw_depth_msaa.create_view(&Default::default());
    let color_view = textures.color.create_view(&Default::default());
    let depth_view = textures.depth_byte.create_view(&Default::default());
    let normal_view = textures.normal.create_view(&Default::default());
    let occupancy_view = textures.occupancy.create_view(&Default::default());
    let raw_depth_view = textures.raw_depth.create_view(&Default::default());
    let gbuffer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh-projector MSAA G-buffer resolve bind group"),
        layout: &pipeline_resources.resolve.gbuffer_layout,
        entries: &[
            texture_bind_entry(0, &color_msaa_view),
            texture_bind_entry(1, &depth_msaa_view),
            texture_bind_entry(2, &normal_msaa_view),
            texture_bind_entry(3, &occupancy_msaa_view),
            texture_bind_entry(4, &raw_depth_msaa_view),
            storage_texture_bind_entry(5, &color_view),
            storage_texture_bind_entry(6, &depth_view),
            storage_texture_bind_entry(7, &normal_view),
            storage_texture_bind_entry(8, &occupancy_view),
            bind_entry(9, &params_buffer),
        ],
    });
    let workgroups = (resolution as u32).div_ceil(IMAGE_WORKGROUP_SIZE);
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mesh-projector MSAA G-buffer resolve pass"),
            ..Default::default()
        });
        pass.set_pipeline(&pipeline_resources.resolve.gbuffer_pipeline);
        pass.set_bind_group(0, &gbuffer_bind_group, &[]);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }

    let raw_depth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh-projector MSAA raw-depth resolve bind group"),
        layout: &pipeline_resources.resolve.raw_depth_layout,
        entries: &[
            texture_bind_entry(0, &raw_depth_msaa_view),
            texture_bind_entry(1, &occupancy_msaa_view),
            storage_texture_bind_entry(2, &raw_depth_view),
            bind_entry(3, &params_buffer),
        ],
    });
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("mesh-projector MSAA raw-depth resolve pass"),
        ..Default::default()
    });
    pass.set_pipeline(&pipeline_resources.resolve.raw_depth_pipeline);
    pass.set_bind_group(0, &raw_depth_bind_group, &[]);
    pass.dispatch_workgroups(workgroups, workgroups, 1);
}

pub(crate) fn encode_pack_outputs(
    device: &wgpu::Device,
    pipeline_resources: &ImagePipelineResources,
    encoder: &mut wgpu::CommandEncoder,
    textures: &RenderTextures,
    resolution: u16,
    view_index: usize,
    selection: WgpuOutputSelection,
    use_ssao_color: bool,
    buffers: &PackedOutputGpuBuffers,
) {
    let color_texture = if use_ssao_color {
        textures.ssao_color.as_ref().unwrap_or(&textures.color)
    } else {
        &textures.color
    };
    let params = GpuPackParams {
        resolution: resolution as u32,
        pixels_per_view: resolution as u32 * resolution as u32,
        view_index: view_index as u32,
        selection_bits: output_selection_bits(selection),
        dispatch_width: pack_dispatch_width(resolution),
        color_stride_words: buffers.color_stride_words,
        scalar_words_per_view: buffers.scalar_words_per_view,
        scalar_stride_words: buffers.scalar_stride_words,
    };
    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh-projector pack params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let color_view = color_texture.create_view(&Default::default());
    let depth_view = textures.depth_byte.create_view(&Default::default());
    let normal_view = textures.normal.create_view(&Default::default());
    let occupancy_view = textures.occupancy.create_view(&Default::default());
    let color_offset = plane_binding_offset(&buffers.color, view_index);
    let depth_offset = plane_binding_offset(&buffers.depth, view_index);
    let normal_offset = plane_binding_offset(&buffers.normal, view_index);
    let occupancy_offset = plane_binding_offset(&buffers.occupancy, view_index);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mesh-projector pack bind group"),
        layout: &pipeline_resources.pack.layout,
        entries: &[
            texture_bind_entry(0, &color_view),
            texture_bind_entry(1, &depth_view),
            texture_bind_entry(2, &normal_view),
            texture_bind_entry(3, &occupancy_view),
            bind_entry_range(
                4,
                &buffers.color.storage,
                color_offset,
                buffers.color.stride_bytes,
            ),
            bind_entry_range(
                5,
                &buffers.depth.storage,
                depth_offset,
                buffers.depth.stride_bytes,
            ),
            bind_entry_range(
                6,
                &buffers.normal.storage,
                normal_offset,
                buffers.normal.stride_bytes,
            ),
            bind_entry_range(
                7,
                &buffers.occupancy.storage,
                occupancy_offset,
                buffers.occupancy.stride_bytes,
            ),
            bind_entry(8, &params_buffer),
        ],
    });
    let pixels = resolution as u32 * resolution as u32;
    let workgroups_x = pack_workgroups_x(resolution);
    let dispatch_width = workgroups_x * 256;
    let workgroups_y = pixels.div_ceil(dispatch_width);
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("mesh-projector pack pass"),
        ..Default::default()
    });
    pass.set_pipeline(&pipeline_resources.pack.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

fn plane_binding_offset(
    plane: &crate::image::resources::PackedOutputPlaneBuffer,
    view_index: usize,
) -> u64 {
    if plane.bytes_per_view == 0 {
        0
    } else {
        view_index as u64 * plane.stride_bytes
    }
}

fn output_selection_bits(selection: WgpuOutputSelection) -> u32 {
    (selection.color as u32)
        | ((selection.depth as u32) << 1)
        | ((selection.normal as u32) << 2)
        | ((selection.occupancy as u32) << 3)
}

fn pack_workgroups_x(resolution: u16) -> u32 {
    let pixels = resolution as u32 * resolution as u32;
    pixels.div_ceil(256).min(256).max(1)
}

fn pack_dispatch_width(resolution: u16) -> u32 {
    pack_workgroups_x(resolution) * 256
}
