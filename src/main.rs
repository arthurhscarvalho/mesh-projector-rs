use mesh_projector;
use mesh_projector::logging::atomic_init_logger;
use mesh_projector::projector::config::ProjectionConfig;
use mesh_projector::projector::config::RotationConfig;
use mesh_projector::types::orientation::Orientation;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::ParallelIterator;
use std::path::Path;

fn main() {
    // Initialize logger
    atomic_init_logger(true).unwrap();
    // Set up mesh path, precision, filtering radius, and output directory
    const MESH_PATH: &str = "assets/just_a_girl_C0-L5_deq_tri.obj";
    const PRECISION: u64 = 12;
    const FILTERING_RADIUS: u64 = 0;
    let output_dir: &Path = Path::new("projections/");
    // Read mesh from OBJ file
    let mesh = mesh_projector::mesh_io::read_mesh(MESH_PATH, None, 12, true, false).unwrap();
    // Set up rotation config for cube projection (will be handled internally)
    let rotation_config = RotationConfig {
        angle_deltas: Orientation::from([0.0, 0.0, 0.0]),
        num_steps: 1,
    };
    // Set up projection settings for cube projection
    let settings = ProjectionConfig {
        precision: PRECISION,
        filtering_window_radius: FILTERING_RADIUS,
        rotation_config,
    };
    // Project to image on bounding cube
    let projections =
        mesh_projector::projector::project_to_image_on_bounding_cube(&mesh, &settings);
    // Save projections
    projections
        .par_iter()
        .enumerate()
        .for_each(|(i, (projection, _, _, _))| {
            mesh_projector::mesh_io::save_image_data(&projection.to_owned(), output_dir, i)
                .unwrap();
        });
}
