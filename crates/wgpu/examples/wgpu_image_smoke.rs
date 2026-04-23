use mesh_projector_core::config::{
    BackgroundConfig, CameraConfig, MsaaConfig, MsaaSampleCount, ProjectionConfig,
};
use mesh_projector_core::config::{LightComponents, LightSource, LightingConfig, SsaoConfig};
use mesh_projector_core::logging::atomic_init_logger;
use mesh_projector_core::mesh_io::{build_preprocessing_from_scene, preprocessed_scene};
use mesh_projector_core::types::color::Color;
use mesh_projector_core::types::orientation::Orientation;
use mesh_projector_wgpu::WgpuProjector;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = atomic_init_logger(false);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let ssao = if args.iter().any(|arg| arg == "--ssao") {
        SsaoConfig::default()
    } else {
        SsaoConfig::disabled()
    };
    let msaa = parse_msaa(&args)?;

    let mesh = mesh_projector_core::mesh_io::read_mesh(
        "assets/just_a_girl_C0-L5_deq_tri.obj",
        None,
        256,
        true,
        false,
    )?;
    let preprocessing = build_preprocessing_from_scene(&mesh, 256);
    let mesh = preprocessed_scene(&mesh, &preprocessing);
    let settings = ProjectionConfig {
        resolution: 256,
        filtering_window_radius: 0,
        view_angles: vec![Orientation::new(0.0, 0.0, 0.0)],
        camera: CameraConfig::default(),
        msaa: MsaaConfig { sample_count: msaa },
        ssao,
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

    let projector = WgpuProjector::new_blocking()?;
    let batch = projector.project_to_image(&mesh, &settings, &lighting)?;
    let occupied = batch
        .occupancy_maps
        .iter()
        .filter(|value| **value == 255)
        .count();
    println!(
        "adapter={:?} images={} occupied={}",
        projector.adapter_info().backend,
        batch.images.shape()[0],
        occupied
    );
    Ok(())
}

fn parse_msaa(args: &[String]) -> Result<MsaaSampleCount, Box<dyn std::error::Error>> {
    let Some(index) = args.iter().position(|arg| arg == "--msaa") else {
        return Ok(MsaaSampleCount::X1);
    };
    let Some(value) = args.get(index + 1) else {
        return Err("--msaa requires one of: 1, 2, 4, 8".into());
    };
    match value.as_str() {
        "1" => Ok(MsaaSampleCount::X1),
        "2" => Ok(MsaaSampleCount::X2),
        "4" => Ok(MsaaSampleCount::X4),
        "8" => Ok(MsaaSampleCount::X8),
        _ => Err("--msaa requires one of: 1, 2, 4, 8".into()),
    }
}
