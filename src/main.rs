use mesh_projector;
use mesh_projector::logging::atomic_init_logger;
use mesh_projector::projector::config::ProjectionConfig;
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
    const VIEW_ANGLES: [[f32; 3]; 6] = [
        [0.0, 0.0, 0.0],       // Front: Cam at +Z, looking -Z. No rotation.
        [0.0, PI, 0.0],        // Back: Cam at -Z, looking +Z. Rotate 180 deg around Y.
        [0.0, PI / 2.0, 0.0],  // Right: Cam at +X, looking -X. Rotate 90 deg around Y.
        [0.0, -PI / 2.0, 0.0], // Left: Cam at -X, looking +X. Rotate -90 deg around Y.
        [PI / 2.0, 0.0, 0.0],  // Top: Cam at +Y, looking -Y. Rotate 90 deg around X.
        [-PI / 2.0, 0.0, 0.0], // Bottom: Cam at -Y, looking +Y. Rotate -90 deg around X.
    ];
    let view_angles = VIEW_ANGLES
        .iter()
        .map(|angle| Orientation::from(*angle))
        .collect();
    // Read mesh from OBJ file
    let mesh =
        mesh_projector::mesh_io::read_mesh(MESH_PATH, None, RESOLUTION, true, false).unwrap();
    // Set up projection settings for cube projection
    let settings = ProjectionConfig {
        resolution: RESOLUTION,
        filtering_window_radius: FILTERING_RADIUS,
        view_angles,
    };
    // Project to image on bounding cube
    let projections = mesh_projector::projector::project_to_image(&mesh, &settings);
    // Save projections
    projections
        .par_iter()
        .enumerate()
        .for_each(|(i, (projection, _, _, _))| {
            mesh_projector::mesh_io::save_image_data(&projection.to_owned(), output_dir, i)
                .unwrap();
        });
}
