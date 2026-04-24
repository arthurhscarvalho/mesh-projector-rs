use crate::error::{WgpuProjectorError, WgpuResult};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

pub(crate) fn create_storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    contents: &[u8],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::STORAGE,
    })
}

pub(crate) fn create_readback_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: wgpu::BufferAddress,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

pub(crate) fn read_buffer(instance: &wgpu::Instance, buffer: &wgpu::Buffer) -> WgpuResult<Vec<u8>> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    instance.poll_all(true);
    receiver
        .recv()
        .map_err(|_| WgpuProjectorError::BufferMapChannelClosed)?
        .map_err(WgpuProjectorError::BufferMap)?;
    let view = slice.get_mapped_range();
    let bytes = view.to_vec();
    drop(view);
    buffer.unmap();
    Ok(bytes)
}
