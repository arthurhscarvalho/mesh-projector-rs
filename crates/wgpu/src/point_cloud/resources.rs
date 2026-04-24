use std::sync::{Arc, Mutex};

const READBACK_RING_SLOTS: usize = 3;

pub(crate) struct PointCloudOutputBuffers {
    pub(crate) positions: wgpu::Buffer,
    pub(crate) colors: wgpu::Buffer,
    pub(crate) positions_size: u64,
    pub(crate) colors_size: u64,
}

#[derive(Default)]
pub(crate) struct PointCloudReadbackPool {
    slots: Vec<ReadbackSlot>,
    next_slot: usize,
}

#[derive(Default)]
struct ReadbackSlot {
    in_use: bool,
    positions: Option<PooledReadbackBuffer>,
    colors: Option<PooledReadbackBuffer>,
}

struct PooledReadbackBuffer {
    buffer: wgpu::Buffer,
    size: u64,
}

pub(crate) struct PointCloudReadbackLease {
    slot_index: usize,
    pub(crate) positions: wgpu::Buffer,
    pub(crate) colors: wgpu::Buffer,
}

pub(crate) type PointCloudReadbackPoolHandle = Arc<Mutex<PointCloudReadbackPool>>;

pub(crate) struct PointCloudReadbackLeaseGuard {
    pool: PointCloudReadbackPoolHandle,
    lease: Option<PointCloudReadbackLease>,
    mapped: bool,
}

impl PointCloudReadbackLeaseGuard {
    pub(crate) fn new(pool: PointCloudReadbackPoolHandle, lease: PointCloudReadbackLease) -> Self {
        Self {
            pool,
            lease: Some(lease),
            mapped: false,
        }
    }

    pub(crate) fn lease(&self) -> &PointCloudReadbackLease {
        self.lease
            .as_ref()
            .expect("point cloud readback lease was already released")
    }

    pub(crate) fn mark_mapped(&mut self) {
        self.mapped = true;
    }
}

impl Drop for PointCloudReadbackLeaseGuard {
    fn drop(&mut self) {
        if let Some(lease) = self.lease.take() {
            if self.mapped {
                lease.unmap_all();
            }
            self.pool
                .lock()
                .expect("point cloud readback pool mutex poisoned")
                .release(lease);
        }
    }
}

pub(crate) fn create_point_cloud_output_buffers(
    device: &wgpu::Device,
    num_points: usize,
) -> PointCloudOutputBuffers {
    let positions_size = (num_points * std::mem::size_of::<[f32; 4]>()) as u64;
    let colors_size = (num_points * std::mem::size_of::<u32>()) as u64;
    PointCloudOutputBuffers {
        positions: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-projector point cloud output positions"),
            size: positions_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        colors: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mesh-projector point cloud output colors"),
            size: colors_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }),
        positions_size,
        colors_size,
    }
}

pub(crate) fn enqueue_point_cloud_readback_copies(
    encoder: &mut wgpu::CommandEncoder,
    output: &PointCloudOutputBuffers,
    readback: &PointCloudReadbackLease,
) {
    encoder.copy_buffer_to_buffer(
        &output.positions,
        0,
        &readback.positions,
        0,
        output.positions_size,
    );
    encoder.copy_buffer_to_buffer(&output.colors, 0, &readback.colors, 0, output.colors_size);
}

impl PointCloudReadbackPool {
    pub(crate) fn acquire(
        &mut self,
        device: &wgpu::Device,
        positions_size: u64,
        colors_size: u64,
    ) -> PointCloudReadbackLease {
        let slot_index = self.acquire_slot_index();
        let slot = &mut self.slots[slot_index];
        slot.in_use = true;

        slot.positions = Some(ensure_readback_buffer(
            device,
            slot.positions.take(),
            "mesh-projector point cloud positions readback",
            positions_size,
        ));
        slot.colors = Some(ensure_readback_buffer(
            device,
            slot.colors.take(),
            "mesh-projector point cloud colors readback",
            colors_size,
        ));

        PointCloudReadbackLease {
            slot_index,
            positions: slot.positions.as_ref().unwrap().buffer.clone(),
            colors: slot.colors.as_ref().unwrap().buffer.clone(),
        }
    }

    pub(crate) fn release(&mut self, lease: PointCloudReadbackLease) {
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

impl PointCloudReadbackLease {
    pub(crate) fn unmap_all(&self) {
        self.positions.unmap();
        self.colors.unmap();
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
