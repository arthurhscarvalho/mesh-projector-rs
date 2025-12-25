use mesh_projector;
use mesh_projector::logging::atomic_init_logger;
use mesh_projector::projector::config::{
    CameraConfig, PerspectiveCameraConfig, ProjectionConfig, ProjectionType,
};
use mesh_projector::renderer::config::{LightComponents, LightSource, LightingConfig};
use mesh_projector::types::color::Color;
use mesh_projector::types::coordinate::Coordinate;
use mesh_projector::types::orientation::Orientation;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use std::path::Path;

fn main() {
    // Initialize logger
    atomic_init_logger(true).unwrap();
    // Set up mesh path, resolution, filtering radius, and output directory
    const MESH_PATH: &str = "assets/just_a_girl_C0-L5_deq_tri.obj";
    const RESOLUTION: u16 = 4096;
    const FILTERING_RADIUS: u64 = 0;
    let output_dir: &Path = Path::new("projections/");
    // Set up view angles
    const PI: f32 = std::f32::consts::PI;
    let view_angles = vec![
        Orientation::new(0.0, 0.0, 0.0), // Front: Cam at +Z, looking -Z. No rotation.
        Orientation::new(0.0, PI, 0.0),  // Back: Cam at -Z, looking +Z. Rotate 180 deg around Y.
        Orientation::new(0.0, PI / 2.0, 0.0), // Right: Cam at +X, looking -X. Rotate 90 deg around Y.
        Orientation::new(0.0, -PI / 2.0, 0.0), // Left: Cam at -X, looking +X. Rotate -90 deg around Y.
        Orientation::new(PI / 2.0, 0.0, 0.0), // Top: Cam at +Y, looking -Y. Rotate 90 deg around X.
        Orientation::new(-PI / 2.0, 0.0, 0.0), // Bottom: Cam at -Y, looking +Y. Rotate -90 deg around X.
    ];
    // Read mesh from OBJ file
    let mesh =
        mesh_projector::mesh_io::read_mesh(MESH_PATH, None, RESOLUTION, true, false).unwrap();
    // Set up projection settings for cube projection
    let settings = ProjectionConfig {
        resolution: RESOLUTION,
        filtering_window_radius: FILTERING_RADIUS,
        view_angles,
        camera: CameraConfig {
            projection: ProjectionType::Perspective(PerspectiveCameraConfig {
                vertical_fov_degrees: 70.0,
            }),
        },
    };
    // Set up lighting configuration
    let lighting_config = LightingConfig {
        source: LightSource::Point(vec![
            Coordinate::new(1.0, 1.0, 1.0),
            Coordinate::new(-1.0, 1.0, 1.0),
            Coordinate::new(-1.0, 1.0, -1.0),
            Coordinate::new(1.0, 1.0, -1.0),
        ]),
        components: LightComponents {
            global_ambient: Color::from_srgb(0.4, 0.4, 0.4),
            global_diffuse: Color::from_srgb(0.7, 0.7, 0.7),
            global_specular: Color::from_srgb(0.0, 0.0, 0.0),
        },
    };
    // Project to image on bounding cube
    let projections =
        mesh_projector::projector::project_to_image(&mesh, &settings, &lighting_config);
    // Save projections
    projections.par_iter().enumerate().for_each(
        |(i, (projection, depth_map, normal_map, occupancy_map))| {
            let path = output_dir.join(format!("projection_{}.png", i));
            mesh_projector::mesh_io::save_image_data(projection.into_dyn(), &path).unwrap();
            let path = output_dir.join(format!("depth_map_{}.png", i));
            mesh_projector::mesh_io::save_image_data(depth_map.into_dyn(), &path).unwrap();
            let path = output_dir.join(format!("normal_map_{}.png", i));
            mesh_projector::mesh_io::save_image_data(normal_map.into_dyn(), &path).unwrap();
            let path = output_dir.join(format!("occupancy_map_{}.png", i));
            mesh_projector::mesh_io::save_image_data(occupancy_map.into_dyn(), &path).unwrap();
        },
    );
}
