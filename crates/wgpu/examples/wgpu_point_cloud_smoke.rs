use mesh_projector_core::config::PointCloudConfig;
use mesh_projector_core::config::{LightComponents, LightSource, LightingConfig};
use mesh_projector_core::logging::atomic_init_logger;
use mesh_projector_core::preprocessing::{build_preprocessing_from_scene, preprocessed_scene};
use mesh_projector_core::types::color::Color;
use mesh_projector_wgpu::WgpuProjector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = atomic_init_logger(false);

    let mesh = mesh_projector_core::mesh_io::read_mesh(
        "assets/just_a_girl_C0-L5_deq_tri.obj",
        None,
        256,
        true,
        false,
    )?;
    let preprocessing = build_preprocessing_from_scene(&mesh, 256);
    let mesh = preprocessed_scene(&mesh, &preprocessing);
    let settings = PointCloudConfig {
        resolution: 256,
        point_cloud_density_factor: 0.001,
    };
    let lighting = LightingConfig {
        source: LightSource::Ambient,
        components: LightComponents {
            global_ambient: Color::white(),
            global_diffuse: Color::white(),
            global_specular: Color::black(),
        },
    };

    let projector = WgpuProjector::new_blocking()?;
    let (points, colors) =
        projector.project_to_point_cloud_by_sampling(&mesh, &settings, &lighting)?;
    println!(
        "adapter={:?} points={} colors={}",
        projector.adapter_info().backend,
        points.nrows(),
        colors.nrows()
    );
    Ok(())
}
