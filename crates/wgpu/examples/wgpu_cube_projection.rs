use mesh_projector_core::config::{
    BackgroundConfig, CameraConfig, MsaaConfig, MsaaSampleCount, ProjectionConfig,
};
use mesh_projector_core::config::{LightComponents, LightSource, LightingConfig, SsaoConfig};
use mesh_projector_core::logging::atomic_init_logger;
use mesh_projector_core::mesh_io::{build_preprocessing_from_scene, preprocessed_scene};
use mesh_projector_core::types::color::Color;
use mesh_projector_core::types::coordinate::Coordinate;
use mesh_projector_core::types::orientation::Orientation;
use mesh_projector_wgpu::{WgpuOutputSelection, WgpuProjector};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    atomic_init_logger(true)?;

    const MESH_PATH: &str = "assets/just_a_girl_C0-L5_deq_tri.obj";
    const RESOLUTION: u16 = 4096;
    let output_dir = Path::new("projections_wgpu/");
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("Usage: cargo run --release --example wgpu_cube_projection -- [--all-maps|--color-only|--color-occupancy] [--msaa 1|2|4|8] [--ssao] [--prepared-repeat N]");
        return Ok(());
    }
    let selection = if args.iter().any(|arg| arg == "--color-only") {
        WgpuOutputSelection::color_only()
    } else if args.iter().any(|arg| arg == "--color-occupancy") {
        WgpuOutputSelection::color_and_occupancy()
    } else {
        WgpuOutputSelection::all()
    };
    let msaa = parse_msaa(&args)?;
    let ssao = if args.iter().any(|arg| arg == "--ssao") {
        SsaoConfig::default()
    } else {
        SsaoConfig::disabled()
    };
    let prepared_repeat = parse_prepared_repeat(&args)?;

    let pi = std::f32::consts::PI;
    let view_angles = vec![
        Orientation::new(0.0, 0.0, 0.0),
        Orientation::new(0.0, pi, 0.0),
        Orientation::new(0.0, pi / 2.0, 0.0),
        Orientation::new(0.0, -pi / 2.0, 0.0),
        Orientation::new(pi / 2.0, 0.0, 0.0),
        Orientation::new(-pi / 2.0, 0.0, 0.0),
    ];

    let read_started = Instant::now();
    let mesh = mesh_projector_core::mesh_io::read_mesh(MESH_PATH, None, RESOLUTION, true, false)?;
    let preprocessing = build_preprocessing_from_scene(&mesh, RESOLUTION);
    let mesh = preprocessed_scene(&mesh, &preprocessing);
    let read_elapsed = read_started.elapsed();
    let settings = ProjectionConfig {
        resolution: RESOLUTION,
        filtering_window_radius: 0,
        view_angles,
        camera: CameraConfig::default(),
        msaa: MsaaConfig { sample_count: msaa },
        ssao,
        background: BackgroundConfig::default(),
    };

    let lighting_config = LightingConfig {
        source: LightSource::Point(vec![
            Coordinate::new(-1.0, 1.0, 1.0),
            Coordinate::new(1.0, 1.0, -1.0),
        ]),
        components: LightComponents {
            global_ambient: Color::from_srgb(0.6, 0.6, 0.6),
            global_diffuse: Color::from_srgb(0.6, 0.6, 0.6),
            global_specular: Color::from_srgb(0.0, 0.0, 0.0),
        },
    };

    let projector = WgpuProjector::new_blocking()?;
    println!("Using adapter: {:?}", projector.adapter_info());
    println!(
        "Supported GPU MSAA samples: {:?}",
        projector.supported_msaa_sample_counts()
    );
    println!("Mesh reading time: {:.3?}", read_elapsed);
    println!(
        "Requested GPU outputs: {}",
        output_selection_label(selection)
    );
    println!(
        "MSAA samples: {}, SSAO: {}, prepared repeats: {}",
        settings.msaa.sample_count.samples(),
        settings.ssao.enabled,
        prepared_repeat
    );

    let compute_started = Instant::now();
    let prepared_started = Instant::now();
    let prepared = projector.prepare_scene(&mesh)?;
    let prepare_elapsed = prepared_started.elapsed();
    println!("Prepared scene setup time: {:.3?}", prepare_elapsed);
    let mut packed = None;
    let mut timings = None;
    for repeat_index in 0..prepared_repeat {
        let (repeat_packed, mut repeat_timings) = projector
            .project_prepared_to_packed_image_with_timings(
                &prepared,
                &settings,
                &lighting_config,
                selection,
            )?;
        if repeat_index == 0 {
            repeat_timings.prepare_scene = prepare_elapsed;
            repeat_timings.setup += prepare_elapsed;
        }
        packed = Some(repeat_packed);
        timings = Some(repeat_timings);
    }
    let packed = packed.expect("prepared repeat count is at least one");
    let timings = timings.expect("prepared repeat count is at least one");
    let packed_elapsed = compute_started.elapsed();
    let convert_started = Instant::now();
    let projections = packed.into_image_batch();
    let convert_elapsed = convert_started.elapsed();
    let compute_elapsed = compute_started.elapsed();
    println!("Projection packed-output time: {:.3?}", packed_elapsed);
    println!("ImageBatch conversion time: {:.3?}", convert_elapsed);
    println!(
        "Projection + compatibility conversion time: {:.3?}",
        compute_elapsed
    );
    println!(
        "Projection stages: setup={:.3?}, prepare_scene={:.3?}, cull_indices={:.3?}, upload_indices={:.3?}, render_submit={:.3?}, msaa_resolve={:.3?}, ssao={:.3?}, pack_output={:.3?}, readback={:.3?}, image_batch_conversion={:.3?}",
        timings.setup,
        timings.prepare_scene,
        timings.cull_indices,
        timings.upload_indices,
        timings.render_submit,
        timings.msaa_resolve,
        timings.ssao,
        timings.pack_output,
        timings.readback,
        convert_elapsed
    );

    projections.par_iter().enumerate().for_each(
        |(i, (projection, depth_map, normal_map, occupancy_map))| {
            if selection.color {
                let path = output_dir.join(format!("projection_{}.png", i));
                mesh_projector_core::mesh_io::save_image_data(projection.into_dyn(), &path)
                    .unwrap();
            }

            if selection.depth {
                let path = output_dir.join(format!("depth_map_{}.png", i));
                mesh_projector_core::mesh_io::save_image_data(depth_map.into_dyn(), &path).unwrap();
            }

            if selection.normal {
                let path = output_dir.join(format!("normal_map_{}.png", i));
                mesh_projector_core::mesh_io::save_image_data(normal_map.into_dyn(), &path)
                    .unwrap();
            }

            if selection.occupancy {
                let path = output_dir.join(format!("occupancy_map_{}.png", i));
                mesh_projector_core::mesh_io::save_image_data(occupancy_map.into_dyn(), &path)
                    .unwrap();
            }
        },
    );

    println!("Saved projections to {}", output_dir.display());
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

fn parse_prepared_repeat(args: &[String]) -> Result<usize, Box<dyn std::error::Error>> {
    let Some(index) = args.iter().position(|arg| arg == "--prepared-repeat") else {
        return Ok(1);
    };
    let Some(value) = args.get(index + 1) else {
        return Err("--prepared-repeat requires a positive integer".into());
    };
    let repeats = value.parse::<usize>()?;
    Ok(repeats.max(1))
}

fn output_selection_label(selection: WgpuOutputSelection) -> &'static str {
    match (
        selection.color,
        selection.depth,
        selection.normal,
        selection.occupancy,
    ) {
        (true, true, true, true) => "all maps",
        (true, false, false, false) => "color only",
        (true, false, false, true) => "color + occupancy",
        _ => "custom",
    }
}
