//! Image projection GPU texture and packed output buffer allocation.

use crate::gpu::formats::{texture_extent, DEPTH_FORMAT, R8_FORMAT, RAW_DEPTH_FORMAT, RGBA_FORMAT};
use crate::image::output::WgpuOutputSelection;

const READBACK_RING_SLOTS: usize = 3;

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
    pub(crate) color: PackedOutputPlaneBuffer,
    pub(crate) depth: PackedOutputPlaneBuffer,
    pub(crate) normal: PackedOutputPlaneBuffer,
    pub(crate) occupancy: PackedOutputPlaneBuffer,
    pub(crate) color_stride_words: u32,
    pub(crate) scalar_words_per_view: u32,
    pub(crate) scalar_stride_words: u32,
}

pub(crate) struct PackedOutputPlaneBuffer {
    pub(crate) storage: wgpu::Buffer,
    pub(crate) bytes_per_view: u64,
    pub(crate) stride_bytes: u64,
    pub(crate) total_bytes: u64,
}

#[derive(Default)]
pub(crate) struct WgpuReadbackPool {
    slots: Vec<ReadbackSlot>,
    next_slot: usize,
}

#[derive(Default)]
struct ReadbackSlot {
    in_use: bool,
    color: Option<PooledReadbackBuffer>,
    depth: Option<PooledReadbackBuffer>,
    normal: Option<PooledReadbackBuffer>,
    occupancy: Option<PooledReadbackBuffer>,
}

struct PooledReadbackBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

pub(crate) struct WgpuReadbackLease {
    slot_index: usize,
    pub(crate) color: Option<wgpu::Buffer>,
    pub(crate) depth: Option<wgpu::Buffer>,
    pub(crate) normal: Option<wgpu::Buffer>,
    pub(crate) occupancy: Option<wgpu::Buffer>,
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
    let storage_alignment = device.limits().min_storage_buffer_offset_alignment as u64;
    let color_bytes_per_view = (pixels_per_view * 4) as u64;
    let color_stride_bytes = align_to(color_bytes_per_view, storage_alignment);
    let scalar_words_per_view = pixels_per_view.div_ceil(4) as u64;
    let scalar_bytes_per_view = scalar_words_per_view * 4;
    let scalar_stride_bytes = align_to(scalar_bytes_per_view, storage_alignment);

    PackedOutputGpuBuffers {
        color: create_packed_plane(
            device,
            "mesh-projector packed color",
            selection.color,
            color_bytes_per_view,
            color_stride_bytes,
            num_views,
        ),
        depth: create_packed_plane(
            device,
            "mesh-projector packed depth",
            selection.depth,
            pixels_per_view as u64,
            scalar_stride_bytes,
            num_views,
        ),
        normal: create_packed_plane(
            device,
            "mesh-projector packed normal",
            selection.normal,
            color_bytes_per_view,
            color_stride_bytes,
            num_views,
        ),
        occupancy: create_packed_plane(
            device,
            "mesh-projector packed occupancy",
            selection.occupancy,
            pixels_per_view as u64,
            scalar_stride_bytes,
            num_views,
        ),
        color_stride_words: (color_stride_bytes / 4) as u32,
        scalar_words_per_view: scalar_words_per_view as u32,
        scalar_stride_words: (scalar_stride_bytes / 4) as u32,
    }
}

fn create_packed_plane(
    device: &wgpu::Device,
    label: &'static str,
    selected: bool,
    bytes_per_view: u64,
    stride_bytes: u64,
    num_views: usize,
) -> PackedOutputPlaneBuffer {
    let total_bytes = if selected {
        stride_bytes * num_views as u64
    } else {
        4
    };
    PackedOutputPlaneBuffer {
        storage: create_output_storage_buffer(device, label, total_bytes),
        bytes_per_view: if selected { bytes_per_view } else { 0 },
        stride_bytes: if selected { stride_bytes } else { 4 },
        total_bytes,
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
    readback: &WgpuReadbackLease,
    selection: WgpuOutputSelection,
) {
    if selection.color {
        if let Some(readback) = &readback.color {
            encoder.copy_buffer_to_buffer(
                &buffers.color.storage,
                0,
                readback,
                0,
                buffers.color.total_bytes,
            );
        }
    }
    if selection.depth {
        if let Some(readback) = &readback.depth {
            encoder.copy_buffer_to_buffer(
                &buffers.depth.storage,
                0,
                readback,
                0,
                buffers.depth.total_bytes,
            );
        }
    }
    if selection.normal {
        if let Some(readback) = &readback.normal {
            encoder.copy_buffer_to_buffer(
                &buffers.normal.storage,
                0,
                readback,
                0,
                buffers.normal.total_bytes,
            );
        }
    }
    if selection.occupancy {
        if let Some(readback) = &readback.occupancy {
            encoder.copy_buffer_to_buffer(
                &buffers.occupancy.storage,
                0,
                readback,
                0,
                buffers.occupancy.total_bytes,
            );
        }
    }
}

impl WgpuReadbackPool {
    pub(crate) fn acquire(
        &mut self,
        device: &wgpu::Device,
        buffers: &PackedOutputGpuBuffers,
        selection: WgpuOutputSelection,
    ) -> WgpuReadbackLease {
        let slot_index = self.acquire_slot_index();
        let slot = &mut self.slots[slot_index];
        slot.in_use = true;

        let color = selection.color.then(|| {
            slot.color = Some(ensure_readback_buffer(
                device,
                slot.color.take(),
                "mesh-projector packed color readback",
                buffers.color.total_bytes,
            ));
            slot.color.as_ref().unwrap().buffer.clone()
        });
        let depth = selection.depth.then(|| {
            slot.depth = Some(ensure_readback_buffer(
                device,
                slot.depth.take(),
                "mesh-projector packed depth readback",
                buffers.depth.total_bytes,
            ));
            slot.depth.as_ref().unwrap().buffer.clone()
        });
        let normal = selection.normal.then(|| {
            slot.normal = Some(ensure_readback_buffer(
                device,
                slot.normal.take(),
                "mesh-projector packed normal readback",
                buffers.normal.total_bytes,
            ));
            slot.normal.as_ref().unwrap().buffer.clone()
        });
        let occupancy = selection.occupancy.then(|| {
            slot.occupancy = Some(ensure_readback_buffer(
                device,
                slot.occupancy.take(),
                "mesh-projector packed occupancy readback",
                buffers.occupancy.total_bytes,
            ));
            slot.occupancy.as_ref().unwrap().buffer.clone()
        });

        WgpuReadbackLease {
            slot_index,
            color,
            depth,
            normal,
            occupancy,
        }
    }

    pub(crate) fn release(&mut self, lease: WgpuReadbackLease) {
        if let Some(slot) = self.slots.get_mut(lease.slot_index) {
            slot.in_use = false;
        }
    }

    fn acquire_slot_index(&mut self) -> usize {
        for offset in 0..self.slots.len() {
            let index = (self.next_slot + offset) % self.slots.len().max(1);
            if !self.slots[index].in_use {
                self.next_slot = (index + 1) % READBACK_RING_SLOTS;
                return index;
            }
        }

        let index = self.slots.len();
        self.slots.push(ReadbackSlot::default());
        self.next_slot = (index + 1) % READBACK_RING_SLOTS;
        index
    }
}

fn ensure_readback_buffer(
    device: &wgpu::Device,
    existing: Option<PooledReadbackBuffer>,
    label: &'static str,
    size: u64,
) -> PooledReadbackBuffer {
    if let Some(existing) = existing {
        if existing.size >= size {
            return existing;
        }
    }

    PooledReadbackBuffer {
        buffer: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        size,
    }
}

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        value
    } else {
        value.div_ceil(alignment) * alignment
    }
}

#[cfg(test)]
mod tests {
    use super::align_to;

    #[test]
    fn align_to_rounds_up_to_requested_alignment() {
        assert_eq!(align_to(9, 4), 12);
        assert_eq!(align_to(12, 4), 12);
        assert_eq!(align_to(12, 1), 12);
    }
}
