//! Image projection GPU texture and packed output buffer allocation.

use crate::gpu::buffers;
use crate::gpu::formats::{texture_extent, DEPTH_FORMAT, R8_FORMAT, RAW_DEPTH_FORMAT, RGBA_FORMAT};
use crate::image::output::WgpuOutputSelection;

pub(crate) struct RenderTextures {
    pub(crate) color: wgpu::Texture,
    pub(crate) color_msaa: Option<wgpu::Texture>,
    pub(crate) depth_byte: wgpu::Texture,
    pub(crate) depth_byte_msaa: Option<wgpu::Texture>,
    pub(crate) normal: wgpu::Texture,
    pub(crate) normal_msaa: Option<wgpu::Texture>,
    pub(crate) occupancy: wgpu::Texture,
    pub(crate) occupancy_msaa: Option<wgpu::Texture>,
    pub(crate) raw_depth: wgpu::Texture,
    pub(crate) raw_depth_msaa: Option<wgpu::Texture>,
    pub(crate) ssao_color: Option<wgpu::Texture>,
    pub(crate) depth: wgpu::Texture,
}

pub(crate) struct PackedOutputGpuBuffers {
    pub(crate) views: Vec<PackedOutputGpuViewBuffers>,
    pub(crate) color_bytes_per_view: u64,
    pub(crate) scalar_bytes_per_view: u64,
}

pub(crate) struct PackedOutputGpuViewBuffers {
    pub(crate) color: wgpu::Buffer,
    pub(crate) depth: wgpu::Buffer,
    pub(crate) normal: wgpu::Buffer,
    pub(crate) occupancy: wgpu::Buffer,
    pub(crate) color_readback: Option<wgpu::Buffer>,
    pub(crate) depth_readback: Option<wgpu::Buffer>,
    pub(crate) normal_readback: Option<wgpu::Buffer>,
    pub(crate) occupancy_readback: Option<wgpu::Buffer>,
}

pub(crate) fn create_render_textures(
    device: &wgpu::Device,
    resolution: u16,
    sample_count: u32,
) -> RenderTextures {
    let scalar_resolved_format = if sample_count > 1 {
        RGBA_FORMAT
    } else {
        R8_FORMAT
    };
    RenderTextures {
        color: create_render_texture(device, "mesh-projector color", resolution, RGBA_FORMAT, 1),
        color_msaa: (sample_count > 1).then(|| {
            create_render_texture(
                device,
                "mesh-projector color msaa",
                resolution,
                RGBA_FORMAT,
                sample_count,
            )
        }),
        depth_byte: create_render_texture(
            device,
            "mesh-projector depth byte",
            resolution,
            scalar_resolved_format,
            1,
        ),
        depth_byte_msaa: (sample_count > 1).then(|| {
            create_render_texture(
                device,
                "mesh-projector depth byte msaa",
                resolution,
                R8_FORMAT,
                sample_count,
            )
        }),
        normal: create_render_texture(device, "mesh-projector normal", resolution, RGBA_FORMAT, 1),
        normal_msaa: (sample_count > 1).then(|| {
            create_render_texture(
                device,
                "mesh-projector normal msaa",
                resolution,
                RGBA_FORMAT,
                sample_count,
            )
        }),
        occupancy: create_render_texture(
            device,
            "mesh-projector occupancy",
            resolution,
            scalar_resolved_format,
            1,
        ),
        occupancy_msaa: (sample_count > 1).then(|| {
            create_render_texture(
                device,
                "mesh-projector occupancy msaa",
                resolution,
                R8_FORMAT,
                sample_count,
            )
        }),
        raw_depth: create_render_texture(
            device,
            "mesh-projector raw view depth",
            resolution,
            RAW_DEPTH_FORMAT,
            1,
        ),
        raw_depth_msaa: (sample_count > 1).then(|| {
            create_render_texture(
                device,
                "mesh-projector raw view depth msaa",
                resolution,
                RAW_DEPTH_FORMAT,
                sample_count,
            )
        }),
        ssao_color: None,
        depth: create_depth_texture(
            device,
            "mesh-projector hardware depth",
            resolution,
            sample_count,
        ),
    }
}

pub(crate) fn create_render_texture(
    device: &wgpu::Device,
    label: &'static str,
    resolution: u16,
    format: wgpu::TextureFormat,
    sample_count: u32,
) -> wgpu::Texture {
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::TEXTURE_BINDING;
    if sample_count == 1 && format != R8_FORMAT {
        usage |= wgpu::TextureUsages::STORAGE_BINDING;
    }
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: texture_extent(resolution),
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn create_depth_texture(
    device: &wgpu::Device,
    label: &'static str,
    resolution: u16,
    sample_count: u32,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: texture_extent(resolution),
        mip_level_count: 1,
        sample_count,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    })
}

pub(crate) fn create_packed_output_buffers(
    device: &wgpu::Device,
    num_views: usize,
    pixels_per_view: usize,
    selection: WgpuOutputSelection,
) -> PackedOutputGpuBuffers {
    let color_bytes_per_view = (pixels_per_view * 4) as u64;
    let scalar_words_per_view = pixels_per_view.div_ceil(4);
    let scalar_bytes_per_view = (scalar_words_per_view * 4) as u64;
    let mut views = Vec::with_capacity(num_views);
    for _ in 0..num_views {
        let color_size = if selection.color {
            color_bytes_per_view
        } else {
            4
        };
        let depth_size = if selection.depth {
            scalar_bytes_per_view
        } else {
            4
        };
        let normal_size = if selection.normal {
            color_bytes_per_view
        } else {
            4
        };
        let occupancy_size = if selection.occupancy {
            scalar_bytes_per_view
        } else {
            4
        };
        let color = create_output_storage_buffer(device, "mesh-projector packed color", color_size);
        let depth = create_output_storage_buffer(device, "mesh-projector packed depth", depth_size);
        let normal =
            create_output_storage_buffer(device, "mesh-projector packed normal", normal_size);
        let occupancy =
            create_output_storage_buffer(device, "mesh-projector packed occupancy", occupancy_size);
        let color_readback = selection.color.then(|| {
            buffers::create_readback_buffer(
                device,
                "mesh-projector packed color readback",
                color_bytes_per_view,
            )
        });
        let depth_readback = selection.depth.then(|| {
            buffers::create_readback_buffer(
                device,
                "mesh-projector packed depth readback",
                scalar_bytes_per_view,
            )
        });
        let normal_readback = selection.normal.then(|| {
            buffers::create_readback_buffer(
                device,
                "mesh-projector packed normal readback",
                color_bytes_per_view,
            )
        });
        let occupancy_readback = selection.occupancy.then(|| {
            buffers::create_readback_buffer(
                device,
                "mesh-projector packed occupancy readback",
                scalar_bytes_per_view,
            )
        });
        views.push(PackedOutputGpuViewBuffers {
            color,
            depth,
            normal,
            occupancy,
            color_readback,
            depth_readback,
            normal_readback,
            occupancy_readback,
        });
    }
    PackedOutputGpuBuffers {
        views,
        color_bytes_per_view,
        scalar_bytes_per_view,
    }
}

fn create_output_storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

pub(crate) fn enqueue_packed_output_copies(
    encoder: &mut wgpu::CommandEncoder,
    buffers: &PackedOutputGpuBuffers,
    selection: WgpuOutputSelection,
) {
    for view in &buffers.views {
        if selection.color {
            if let Some(readback) = &view.color_readback {
                encoder.copy_buffer_to_buffer(
                    &view.color,
                    0,
                    readback,
                    0,
                    buffers.color_bytes_per_view,
                );
            }
        }
        if selection.depth {
            if let Some(readback) = &view.depth_readback {
                encoder.copy_buffer_to_buffer(
                    &view.depth,
                    0,
                    readback,
                    0,
                    buffers.scalar_bytes_per_view,
                );
            }
        }
        if selection.normal {
            if let Some(readback) = &view.normal_readback {
                encoder.copy_buffer_to_buffer(
                    &view.normal,
                    0,
                    readback,
                    0,
                    buffers.color_bytes_per_view,
                );
            }
        }
        if selection.occupancy {
            if let Some(readback) = &view.occupancy_readback {
                encoder.copy_buffer_to_buffer(
                    &view.occupancy,
                    0,
                    readback,
                    0,
                    buffers.scalar_bytes_per_view,
                );
            }
        }
    }
}
