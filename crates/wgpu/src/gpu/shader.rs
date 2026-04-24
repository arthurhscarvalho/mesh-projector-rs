pub(crate) const IMAGE_PROJECT: &str = include_str!("../shaders/image_project.wgsl");
pub(crate) const IMAGE_MSAA_RESOLVE: &str = include_str!("../shaders/image_msaa_resolve.wgsl");
pub(crate) const IMAGE_MSAA_RAW_DEPTH_RESOLVE: &str =
    include_str!("../shaders/image_msaa_raw_depth_resolve.wgsl");
pub(crate) const IMAGE_PACK: &str = include_str!("../shaders/image_pack.wgsl");
pub(crate) const IMAGE_SSAO: &str = include_str!("../shaders/image_ssao.wgsl");
pub(crate) const POINT_CLOUD: &str = include_str!("../shaders/point_cloud.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn msaa_depth_resolve_allows_negative_orthographic_depths() {
        assert!(!IMAGE_MSAA_RESOLVE.contains("raw_depth > 0.0"));
        assert!(!IMAGE_MSAA_RAW_DEPTH_RESOLVE.contains("raw_depth > 0.0"));
        assert!(IMAGE_MSAA_RAW_DEPTH_RESOLVE.contains("occupancy_msaa"));
    }
}
