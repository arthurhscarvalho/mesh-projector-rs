pub struct WgpuPreparedPointCloudScene {
    pub(crate) projector_id: u64,
    pub(crate) triangle_buffer: wgpu::Buffer,
    pub(crate) meta_buffer: wgpu::Buffer,
    pub(crate) texel_buffer: wgpu::Buffer,
    pub(crate) offset_buffer: wgpu::Buffer,
    pub(crate) triangle_count: u32,
    pub(crate) total_points: usize,
}
