extern crate ndarray as nd;

pub mod projector;
mod renderer;

use mesh_projector_core::config::{LightingConfig, PointCloudConfig};
use mesh_projector_core::types::mesh::Scene;
use ndarray::Array2;

pub use projector::{project_to_image, project_to_point_cloud_by_sampling};

pub fn project_to_point_cloud(
    mesh: &Scene,
    settings: &PointCloudConfig,
    lighting_config: &LightingConfig,
) -> (Array2<f32>, Array2<u8>) {
    project_to_point_cloud_by_sampling(mesh, settings, lighting_config)
}

#[cfg(test)]
mod tests {
    use super::project_to_image;
    use mesh_projector_core::config::{
        BackgroundConfig, CameraConfig, LightComponents, LightSource, LightingConfig, MsaaConfig,
        MsaaSampleCount, ProjectionConfig, SsaoConfig,
    };
    use mesh_projector_core::types::color::Color;
    use mesh_projector_core::types::mesh::{Scene, ShadingMode};
    use mesh_projector_core::types::orientation::Orientation;

    #[test]
    fn empty_scene_returns_expected_image_batch_shape() {
        let scene = Scene::new(Vec::new(), ShadingMode::VertexColor, None);
        let settings = ProjectionConfig {
            resolution: 8,
            filtering_window_radius: 0,
            view_angles: vec![Orientation::new(0.0, 0.0, 0.0)],
            camera: CameraConfig::default(),
            msaa: MsaaConfig {
                sample_count: MsaaSampleCount::X1,
            },
            ssao: SsaoConfig::disabled(),
            background: BackgroundConfig::default(),
        };
        let lighting = LightingConfig {
            source: LightSource::Ambient,
            components: LightComponents {
                global_ambient: Color::white(),
                global_diffuse: Color::white(),
                global_specular: Color::black(),
            },
        };

        let batch = project_to_image(&scene, &settings, &lighting);

        assert_eq!(batch.images.shape(), &[1, 8, 8, 3]);
        assert_eq!(batch.depth_maps.shape(), &[1, 8, 8]);
        assert!(batch.occupancy_maps.iter().all(|value| *value == 0));
    }
}
