extern crate colog;
extern crate log;
extern crate ndarray as nd;
extern crate num;
extern crate once_cell;
extern crate rayon;

pub mod config;
pub mod constants;
pub mod geometry;
pub mod logging;
pub mod mesh_io;
pub mod types;

#[cfg(test)]
mod tests {
    use crate::config::{
        BackgroundConfig, CameraConfig, MsaaConfig, MsaaSampleCount, ProjectionConfig, SsaoConfig,
    };
    use crate::types::color::Color;
    use crate::types::mesh::{Scene, ShadingMode};
    use crate::types::orientation::Orientation;

    #[test]
    fn config_and_scene_types_construct() {
        let settings = ProjectionConfig {
            resolution: 16,
            filtering_window_radius: 0,
            view_angles: vec![Orientation::new(0.0, 0.0, 0.0)],
            camera: CameraConfig::default(),
            msaa: MsaaConfig {
                sample_count: MsaaSampleCount::X4,
            },
            ssao: SsaoConfig::disabled(),
            background: BackgroundConfig::Rgb(Color::white()),
        };
        let scene = Scene::new(Vec::new(), ShadingMode::VertexColor, None);

        assert_eq!(settings.msaa.sample_count.samples(), 4);
        assert_eq!(settings.background.channels(), 3);
        assert!(scene.meshes.is_empty());
    }
}
