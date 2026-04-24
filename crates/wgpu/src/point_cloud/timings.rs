#[derive(Clone, Copy, Debug, Default)]
pub struct WgpuPointCloudTimings {
    pub total: std::time::Duration,
    pub prepare_scene: std::time::Duration,
    pub upload: std::time::Duration,
    pub compute_submit: std::time::Duration,
    pub copy_to_readback: std::time::Duration,
    pub map_wait: std::time::Duration,
    pub readback: std::time::Duration,
    pub cpu_copy: std::time::Duration,
    pub numpy_conversion: std::time::Duration,
}
