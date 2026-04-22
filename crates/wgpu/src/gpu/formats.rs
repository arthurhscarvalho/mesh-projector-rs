pub(crate) const POINT_CLOUD_WORKGROUP_SIZE: u32 = 128;
pub(crate) const IMAGE_WORKGROUP_SIZE: u32 = 16;
pub(crate) const RGBA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub(crate) const R8_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;
pub(crate) const RAW_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

pub(crate) fn texture_extent(resolution: u16) -> wgpu::Extent3d {
    wgpu::Extent3d {
        width: resolution as u32,
        height: resolution as u32,
        depth_or_array_layers: 1,
    }
}

pub(crate) fn supported_image_msaa_sample_counts(adapter: &wgpu::Adapter) -> Vec<u32> {
    let formats = [RGBA_FORMAT, R8_FORMAT, RAW_DEPTH_FORMAT, DEPTH_FORMAT];
    [1, 2, 4, 8]
        .into_iter()
        .filter(|sample_count| {
            *sample_count == 1
                || formats.iter().all(|format| {
                    adapter
                        .get_texture_format_features(*format)
                        .flags
                        .sample_count_supported(*sample_count)
                })
        })
        .collect()
}
