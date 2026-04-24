use wgpu::util::DeviceExt;

pub(crate) fn create_storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    contents: &[u8],
) -> wgpu::Buffer {
    if contents.is_empty() {
        return device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
    }

    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::STORAGE,
    })
}
