use crate::conditional_log;
use nd::{s, Axis, Ix3, Ix4};
use nd::{Array2, Array3, Array4};
use nd::{ArrayView, ArrayView2, ArrayView3, ArrayViewMut, ArrayViewMut2, ArrayViewMut3};
use rayon::prelude::*;
use std::f32;

// --- Constants ---
const NUM_IMAGES: usize = 6; // 3 axes, 2 directions each (front/back)
const BACKGROUND_BRIGHTNESS: u8 = 255;
const BARYCENTRIC_EPSILON: f32 = 1e-7; // For barycentric denominator check
const BARYCENTRIC_COORD_EPSILON: f32 = -1e-7; // Tolerance for barycentric coordinates (u,v,w) check
const FILTERING_NEIGHBORHOOD_FACTOR: f32 = 0.1; // Factor of max_bound for filter threshold

// --- Data Structures ---

/// Represents texture properties.
#[derive(Debug, Clone)]
pub struct TextureData<'a> {
    pub texture: ArrayView3<'a, u8>,
    pub channels: usize,
    pub height_f32: f32,
    pub width_f32: f32,
}

impl<'a> TextureData<'a> {
    #[inline]
    pub fn new(texture: ArrayView3<'a, u8>) -> Self {
        let shape = texture.shape();
        let height = shape[0];
        let width = shape[1];
        TextureData {
            texture,
            channels: shape[2],
            height_f32: height as f32,
            width_f32: width as f32,
        }
    }
}

/// Holds image buffers: the color images, occlusion maps, and Z-buffers.
#[derive(Debug)]
pub struct Buffers {
    pub images: Array4<u8>,
    pub ocp_maps: Array3<f32>,
    pub z_buffers: Array3<f32>,
}

impl Buffers {
    /// Creates new buffers initialized for the projection process.
    /// Z-buffers are initialized to +/- INFINITY based on view direction
    /// (front views check depth < current_z, back views check depth > current_z).
    pub fn new(num_images: usize, rows: usize, cols: usize, channels: usize) -> Self {
        Buffers {
            images: Array4::from_elem((num_images, rows, cols, channels), BACKGROUND_BRIGHTNESS),
            ocp_maps: Array3::zeros((num_images, rows, cols)),
            z_buffers: {
                let mut z = Array3::from_elem((num_images, rows, cols), f32::INFINITY);
                // Initialize back views (odd indices) with NEG_INFINITY for correct depth testing.
                for i in (1..num_images).step_by(2) {
                    z.slice_mut(s![i, .., ..]).fill(f32::NEG_INFINITY);
                }
                z
            },
        }
    }
}

/// Encapsulates parameters for a single projection view (camera settings).
#[derive(Debug, Clone, Copy)]
pub struct ProjectionConfig {
    /// The axis representing depth (0=X, 1=Y, 2=Z).
    pub depth_axis: usize,
    /// True if this is a front view (positive direction along depth_axis).
    pub is_front_view: bool,
    /// Normalized view direction vector.
    pub view_dir: (f32, f32, f32),
    /// The primary axis index for the 2D projection plane (e.g., Y if depth is X).
    pub p_axis: usize,
    /// The secondary axis index for the 2D projection plane (e.g., Z if depth is X).
    pub q_axis: usize,
}

impl ProjectionConfig {
    /// Creates projection settings based on the image layer index.
    /// Layers 0, 2, 4 are front views (X+, Y+, Z+).
    /// Layers 1, 3, 5 are back views (X-, Y-, Z-).
    pub fn new(img_layer: usize) -> Self {
        let depth_axis = img_layer / 2; // 0 for X, 1 for Y, 2 for Z
        let view_type = img_layer % 2; // 0 for front, 1 for back
        let is_front_view = view_type == 0;

        // Determine view direction based on axis and front/back type
        let view_dir = match (depth_axis, view_type) {
            (0, 0) => (1.0, 0.0, 0.0),  // +X
            (0, 1) => (-1.0, 0.0, 0.0), // -X
            (1, 0) => (0.0, 1.0, 0.0),  // +Y
            (1, 1) => (0.0, -1.0, 0.0), // -Y
            (2, 0) => (0.0, 0.0, 1.0),  // +Z
            (2, 1) => (0.0, 0.0, -1.0), // -Z
            _ => unreachable!("Invalid image layer index: {}", img_layer),
        };

        // Define the axes used for the 2D projection plane (P, Q) based on the depth axis
        // If depth is X (0), plane is YZ (1, 2)
        // If depth is Y (1), plane is XZ (0, 2)
        // If depth is Z (2), plane is XY (0, 1)
        const PROJECTION_PLANE_AXES: [(usize, usize); 3] = [(1, 2), (0, 2), (0, 1)];
        let (p_axis, q_axis) = PROJECTION_PLANE_AXES[depth_axis];

        ProjectionConfig {
            depth_axis,
            is_front_view,
            view_dir,
            p_axis,
            q_axis,
        }
    }
}

// --- Helper Functions ---

/// Computes barycentric coordinates (u, v, w) for point P relative to triangle ABC.
/// Returns None if the triangle is degenerate (denominator is close to zero).
#[inline]
fn barycentric(
    a: (f32, f32), // Point A
    b: (f32, f32), // Point B
    c: (f32, f32), // Point C
    p: (f32, f32), // Point P
) -> Option<(f32, f32, f32)> {
    let v0x = b.0 - a.0;
    let v0y = b.1 - a.1;
    let v1x = c.0 - a.0;
    let v1y = c.1 - a.1;
    let v2x = p.0 - a.0;
    let v2y = p.1 - a.1;

    let d00 = v0x * v0x + v0y * v0y;
    let d01 = v0x * v1x + v0y * v1y;
    let d11 = v1x * v1x + v1y * v1y;
    let d20 = v2x * v0x + v2y * v0y;
    let d21 = v2x * v1x + v2y * v1y;

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < BARYCENTRIC_EPSILON {
        return None; // Degenerate triangle
    }

    let inv_denom = 1.0 / denom;
    let v = (d11 * d20 - d01 * d21) * inv_denom;
    let w = (d00 * d21 - d01 * d20) * inv_denom;
    let u = 1.0 - v - w;

    Some((u, v, w))
}

/// Precomputes the geometric (unnormalized) normal for each triangle face.
fn precompute_triangle_normals(
    points: ArrayView2<f32>,
    triangles: ArrayView2<usize>,
) -> Array2<f32> {
    let num_faces = triangles.nrows();
    let mut triangle_normals = Array2::<f32>::zeros((num_faces, 3));

    // Could be parallelized if num_faces is very large, but often fast enough sequentially.
    for t_idx in 0..num_faces {
        let tri_indices = triangles.row(t_idx);
        let i0 = tri_indices[0];
        let i1 = tri_indices[1];
        let i2 = tri_indices[2];

        let p0 = points.row(i0);
        let p1 = points.row(i1);
        let p2 = points.row(i2);

        // Vectors forming the triangle edges
        let v0x = p1[0] - p0[0];
        let v0y = p1[1] - p0[1];
        let v0z = p1[2] - p0[2];
        let v1x = p2[0] - p0[0];
        let v1y = p2[1] - p0[1];
        let v1z = p2[2] - p0[2];

        // Cross product to find the normal
        let normal_x = v0y * v1z - v0z * v1y;
        let normal_y = v0z * v1x - v0x * v1z;
        let normal_z = v0x * v1y - v0y * v1x;

        triangle_normals[[t_idx, 0]] = normal_x;
        triangle_normals[[t_idx, 1]] = normal_y;
        triangle_normals[[t_idx, 2]] = normal_z;
    }
    triangle_normals
}

/// Checks if a triangle is facing the view direction (basic back-face culling).
#[inline]
fn is_triangle_visible(
    normal_x: f32,
    normal_y: f32,
    normal_z: f32,
    view_dir: (f32, f32, f32),
) -> bool {
    // Dot product of the triangle normal and the view direction.
    // If positive, the normal points towards the viewer (or is orthogonal), so cull it.
    let dot = normal_x * view_dir.0 + normal_y * view_dir.1 + normal_z * view_dir.2;
    dot < 0.0 // Only render if facing away from the view direction (dot product is negative)
}

/// Calculates the 2D bounding box of the projected triangle.
#[inline]
fn calculate_bounding_box(
    p0_proj: (f32, f32), // Projected point 0 (p_axis, q_axis)
    p1_proj: (f32, f32), // Projected point 1
    p2_proj: (f32, f32), // Projected point 2
    rows: usize,
    cols: usize,
) -> Option<(usize, usize, usize, usize)> {
    // (min_row, max_row, min_col, max_col)
    let min_col_f = p0_proj.0.min(p1_proj.0.min(p2_proj.0));
    let max_col_f = p0_proj.0.max(p1_proj.0.max(p2_proj.0));
    let min_row_f = p0_proj.1.min(p1_proj.1.min(p2_proj.1));
    let max_row_f = p0_proj.1.max(p1_proj.1.max(p2_proj.1));

    // Clamp to screen bounds and convert to integer pixel indices
    // Use floor for min and ceil for max to ensure coverage
    let min_col = (min_col_f.floor() as i64).max(0) as usize;
    let max_col = (max_col_f.ceil() as i64).min(cols as i64 - 1).max(0) as usize;
    let min_row = (min_row_f.floor() as i64).max(0) as usize;
    let max_row = (max_row_f.ceil() as i64).min(rows as i64 - 1).max(0) as usize;

    // Check if the bounding box is valid and within screen bounds
    if min_col >= cols || min_row >= rows || min_col > max_col || min_row > max_row {
        None // Bounding box is outside the screen or invalid
    } else {
        Some((min_row, max_row, min_col, max_col))
    }
}

/// Interpolates texture coordinates and samples the texture.
#[inline]
fn sample_texture_color(
    u: f32,                       // Barycentric coordinate for vertex 0
    v: f32,                       // Barycentric coordinate for vertex 1
    w: f32,                       // Barycentric coordinate for vertex 2
    uv0: ArrayView<f32, nd::Ix1>, // UV coords for vertex 0
    uv1: ArrayView<f32, nd::Ix1>, // UV coords for vertex 1
    uv2: ArrayView<f32, nd::Ix1>, // UV coords for vertex 2
    texture_data: &TextureData,
    target_color: &mut ArrayViewMut<u8, nd::Ix1>, // Output color buffer slice (length = channels)
) {
    // Interpolate UV coordinates
    let interp_u_coord = u * uv0[0] + v * uv1[0] + w * uv2[0];
    let interp_v_coord = u * uv0[1] + v * uv1[1] + w * uv2[1];

    // Convert interpolated UVs to texture pixel coordinates (sampling at center)
    // Clamp coordinates to valid texture range.
    // Note: V coordinate is often flipped (1.0 - V) depending on texture convention.
    let tex_x = (interp_u_coord * texture_data.width_f32)
        .round()
        .max(0.0)
        .min(texture_data.width_f32 - 1.0) as usize;
    let tex_y = ((1.0 - interp_v_coord) * texture_data.height_f32) // Assuming V is flipped
        .round()
        .max(0.0)
        .min(texture_data.height_f32 - 1.0) as usize;

    // Sample the texture color at the calculated coordinates
    for chan in 0..texture_data.channels {
        target_color[chan] = texture_data.texture[[tex_y, tex_x, chan]];
    }
}

/// Rasterizes a single triangle onto the appropriate image slices.
fn rasterize_triangle(
    t_idx: usize,                         // Triangle index
    points: ArrayView2<f32>,              // All vertex positions (N, 3)
    triangles: ArrayView2<usize>,         // Triangle vertex indices (M, 3)
    triangle_uvs: ArrayView2<f32>,        // UV coordinates per triangle vertex (M*3, 2)
    triangle_normals: &Array2<f32>,       // Precomputed normals per triangle (M, 3)
    texture_data: &TextureData,           // Texture information
    projection_config: &ProjectionConfig, // View configuration
    z_slice: &mut ArrayViewMut2<f32>,     // Z-buffer for the current view
    image_slice: &mut ArrayViewMut3<u8>,  // Color image buffer for the current view
    ocp_slice: &mut ArrayViewMut2<f32>,   // Occlusion map for the current view
) {
    // 1. Back-face Culling
    let normal = triangle_normals.row(t_idx);
    if !is_triangle_visible(normal[0], normal[1], normal[2], projection_config.view_dir) {
        return;
    }

    // 2. Get Vertex Data (Positions and UVs)
    let tri_indices = triangles.row(t_idx);
    let i0 = tri_indices[0];
    let i1 = tri_indices[1];
    let i2 = tri_indices[2];

    let p0 = points.row(i0);
    let p1 = points.row(i1);
    let p2 = points.row(i2);

    let uv_idx_base = t_idx * 3;
    let uv0 = triangle_uvs.row(uv_idx_base);
    let uv1 = triangle_uvs.row(uv_idx_base + 1);
    let uv2 = triangle_uvs.row(uv_idx_base + 2);

    // 3. Project Vertices to 2D Plane
    let p0_proj = (p0[projection_config.p_axis], p0[projection_config.q_axis]);
    let p1_proj = (p1[projection_config.p_axis], p1[projection_config.q_axis]);
    let p2_proj = (p2[projection_config.p_axis], p2[projection_config.q_axis]);

    // 4. Calculate Bounding Box
    let (rows, cols) = (z_slice.shape()[0], z_slice.shape()[1]);
    let Some((min_row, max_row, min_col, max_col)) =
        calculate_bounding_box(p0_proj, p1_proj, p2_proj, rows, cols)
    else {
        return; // Triangle is completely off-screen
    };

    // 5. Iterate Pixels in Bounding Box
    for r in min_row..=max_row {
        for c in min_col..=max_col {
            // Pixel center coordinates
            let px = c as f32 + 0.5;
            let py = r as f32 + 0.5;

            // 6. Calculate Barycentric Coordinates
            if let Some((u, v, w)) = barycentric(p0_proj, p1_proj, p2_proj, (px, py)) {
                // Check if the pixel center is inside the triangle (with tolerance)
                if u >= BARYCENTRIC_COORD_EPSILON
                    && v >= BARYCENTRIC_COORD_EPSILON
                    && w >= BARYCENTRIC_COORD_EPSILON
                {
                    // 7. Interpolate Depth
                    let interpolated_depth = u * p0[projection_config.depth_axis]
                        + v * p1[projection_config.depth_axis]
                        + w * p2[projection_config.depth_axis];

                    // 8. Depth Test
                    let current_z = z_slice[[r, c]];
                    let depth_test_passed = if projection_config.is_front_view {
                        interpolated_depth < current_z // Closer object has smaller depth
                    } else {
                        interpolated_depth > current_z // Closer object has larger depth (for back view)
                    };

                    if depth_test_passed {
                        // 9. Update Buffers
                        z_slice[[r, c]] = interpolated_depth;
                        ocp_slice[[r, c]] = 1.0; // Mark pixel as covered

                        // 10. Sample Texture and Update Color Buffer
                        let mut color_pixel_slice = image_slice.slice_mut(s![r, c, ..]);
                        sample_texture_color(
                            u,
                            v,
                            w,
                            uv0,
                            uv1,
                            uv2,
                            texture_data,
                            &mut color_pixel_slice,
                        );
                    }
                }
            }
        }
    }
}

/// Finds outlier pixels for a single image layer using parallel processing.
fn find_outliers_for_layer(
    z_slice: ArrayView2<f32>,
    ocp_orig_slice: ArrayView2<f32>, // Use a copy of original occlusion
    w: usize,                        // Filter window radius
    rows: usize,
    cols: usize,
    filter_threshold: f32,
    is_front_view: bool,
) -> Vec<(usize, usize)> {
    let row_start = w;
    let col_start = w;
    let num_inner_rows = rows.saturating_sub(2 * w);
    let num_inner_cols = cols.saturating_sub(2 * w);

    if num_inner_rows == 0 || num_inner_cols == 0 {
        return Vec::new(); // Image too small for filter window
    }

    (0..num_inner_rows)
        .into_par_iter()
        .flat_map(|r_offset| {
            let r = row_start + r_offset;
            let mut row_outlier_coords = Vec::new();
            for c_offset in 0..num_inner_cols {
                let c = col_start + c_offset;

                // Skip background pixels
                if ocp_orig_slice[[r, c]] == 0.0 {
                    continue;
                }

                let center_depth = z_slice[[r, c]];
                if !center_depth.is_finite() {
                    // Skip if depth is Inf/NaN
                    continue;
                }

                // Calculate average depth of valid neighbors
                let mut sum_depth = 0.0;
                let mut count_neighbors = 0.0;
                let r_start_nb = r - w;
                let r_end_nb = r + w + 1;
                let c_start_nb = c - w;
                let c_end_nb = c + w + 1;

                for nr in r_start_nb..r_end_nb {
                    for nc in c_start_nb..c_end_nb {
                        if nr == r && nc == c {
                            continue;
                        } // Skip center pixel

                        // Check if neighbor is part of the object and has valid depth
                        if ocp_orig_slice[[nr, nc]] > 0.0 {
                            let neighbor_depth = z_slice[[nr, nc]];
                            if neighbor_depth.is_finite() {
                                sum_depth += neighbor_depth;
                                count_neighbors += 1.0;
                            }
                        }
                    }
                }

                // Check for outlier condition
                if count_neighbors > 0.0 {
                    let avg_neighbor_depth = sum_depth / count_neighbors;
                    let depth_diff = center_depth - avg_neighbor_depth;

                    // Re-checking original code: it was `depth_diff < -filter_threshold` for back view.
                    let is_outlier_corrected = if is_front_view {
                        depth_diff > filter_threshold // Original was `> filter_threshold`. Center depth is larger (further away). Let's assume original intent was correct.
                    } else {
                        depth_diff < -filter_threshold // Original was `< -filter_threshold`. Center depth is smaller (less negative, so closer to origin, i.e. further away in negative direction)
                    };

                    if is_outlier_corrected {
                        // Use the variable reflecting the original logic
                        row_outlier_coords.push((r, c));
                    }
                }
            }
            row_outlier_coords
        })
        .collect()
}

/// Applies the filter by resetting outlier pixels in the buffers.
fn apply_outlier_filter(
    outlier_coords: &[(usize, usize)],
    mut image_slice: ArrayViewMut3<u8>, // Mutable view to image layer
    mut ocp_slice: ArrayViewMut2<f32>,  // Mutable view to ocp map layer
    channels: usize,
    background_color: u8,
) {
    if outlier_coords.is_empty() {
        return;
    }
    for &(r, c) in outlier_coords {
        ocp_slice[[r, c]] = 0.0; // Mark as background
        for chan in 0..channels {
            image_slice[[r, c, chan]] = background_color; // Reset color
        }
    }
}

/// Filters outlier pixels based on depth differences in a local neighborhood.
fn filter_outliers(
    buffers: &mut Buffers,
    filtering_window_radius: u32, // Renamed from 'filtering' for clarity
    rows: usize,
    cols: usize,
    max_bound_f32: f32, // Used for threshold calculation
    channels: usize,
    background_color: u8,
    logging_enabled: bool,
) -> [u64; NUM_IMAGES] {
    let mut outlier_counts: [u64; NUM_IMAGES] = [0; NUM_IMAGES];

    if filtering_window_radius == 0 {
        conditional_log!(
            logging_enabled,
            info,
            "Filtering skipped: window radius is zero."
        );
        return outlier_counts;
    }

    let w = filtering_window_radius as usize;
    if rows <= 2 * w || cols <= 2 * w {
        conditional_log!(
            logging_enabled,
            warn, // Changed to warn as it might be unexpected
            "Filtering skipped: Image dimensions ({}x{}) too small for filter window radius {}.",
            rows,
            cols,
            w
        );
        return outlier_counts;
    }

    conditional_log!(logging_enabled, info, "Starting outlier filtering loop");
    let filter_threshold = FILTERING_NEIGHBORHOOD_FACTOR * max_bound_f32;

    // Process each layer (view)
    for k in 0..NUM_IMAGES {
        let ocp_orig_slice = buffers.ocp_maps.index_axis(Axis(0), k).to_owned(); // Crucial: Filter based on *original* occlusion
        let z_slice = buffers.z_buffers.index_axis(Axis(0), k);
        let is_front_view = k % 2 == 0;

        // Find outliers in parallel for this layer
        let outlier_coords = find_outliers_for_layer(
            z_slice,
            ocp_orig_slice.view(), // Pass immutable view
            w,
            rows,
            cols,
            filter_threshold,
            is_front_view,
        );

        // Apply the filtering changes to the buffers
        let image_slice = buffers.images.index_axis_mut(Axis(0), k);
        let ocp_slice = buffers.ocp_maps.index_axis_mut(Axis(0), k);
        apply_outlier_filter(
            &outlier_coords,
            image_slice,
            ocp_slice,
            channels,
            background_color,
        );

        outlier_counts[k] = outlier_coords.len() as u64;
    }

    conditional_log!(logging_enabled, info, "Filtering finished");
    outlier_counts
}

// --- Main Function ---

/// Computes 6 orthographic projections (front/back for X/Y/Z) of a textured mesh.
///
/// Args:
/// * `points`: Vertex coordinates (N, 3).
/// * `triangles`: Triangle indices (M, 3).
/// * `texture`: Texture image data (H, W, C).
/// * `triangle_uvs`: UV coordinates for each vertex within each triangle (M * 3, 2).
/// * `precision`: Determines the output image resolution (2^precision x 2^precision).
/// * `filtering`: Radius of the neighborhood window for outlier filtering (0 to disable).
/// * `logging_enabled`: Flag to enable/disable conditional logging.
///
/// Returns:
/// A tuple containing:
/// * `images`: The 6 rendered projection images (6, H, W, C).
/// * `ocp_maps`: The 6 occlusion maps (6, H, W), where 1.0 indicates a mesh pixel.
/// * `freqs`: An array with the count of filtered outlier pixels for each of the 6 images.
pub fn compute_mesh_projections(
    points: ArrayView2<f32>,
    triangles: ArrayView2<usize>,
    texture: ArrayView3<u8>,
    triangle_uvs: ArrayView2<f32>,
    precision: u32,
    filtering_window_radius: u32, // Renamed for clarity
    logging_enabled: bool,
) -> (Array4<u8>, Array3<f32>, [u64; NUM_IMAGES]) {
    conditional_log!(
        logging_enabled,
        info,
        "Initializing optimized projector back-end"
    );

    // --- Setup & Validation ---
    let max_bound: u64 = 1 << precision;
    let max_bound_f32: f32 = max_bound as f32;
    let max_bound_u: usize = max_bound as usize;
    let rows = max_bound_u;
    let cols = max_bound_u;

    let texture_data = TextureData::new(texture);
    let channels = texture_data.channels;

    // Handle empty inputs gracefully
    if points.is_empty() || triangles.is_empty() || texture.is_empty() || triangle_uvs.is_empty() {
        conditional_log!(
            logging_enabled,
            warn,
            "Empty input detected (points, triangles, texture, or uvs). Returning empty buffers."
        );
        let shape4d = Ix4(NUM_IMAGES, rows, cols, channels);
        let shape3d = Ix3(NUM_IMAGES, rows, cols);
        return (
            Array4::from_elem(shape4d, BACKGROUND_BRIGHTNESS),
            Array3::zeros(shape3d),
            [0; NUM_IMAGES],
        );
    }

    // Validate UV buffer size consistency
    if triangles.nrows() * 3 != triangle_uvs.nrows() {
        // Using panic as in the original code. Consider Result for production.
        panic!(
            "Inconsistent input: Number of triangle vertices ({}) != number of triangle UVs ({}). Expected {} UV coordinates.",
            triangles.nrows(),
            triangle_uvs.nrows(),
            triangles.nrows() * 3
        );
    }

    // --- Precomputation ---
    conditional_log!(logging_enabled, info, "Precomputing triangle normals");
    let triangle_normals = precompute_triangle_normals(points, triangles);
    conditional_log!(logging_enabled, info, "Normal precomputation finished");

    // --- Buffer Initialization ---
    let mut buffers = Buffers::new(NUM_IMAGES, rows, cols, channels);

    // --- Rasterization (Parallel over Views) ---
    conditional_log!(
        logging_enabled,
        info,
        "Starting parallel rasterization loop for {} views",
        NUM_IMAGES
    );

    // Prepare mutable views for parallel processing
    // Collecting into Vecs allows `par_iter_mut`
    let mut z_buffer_views: Vec<_> = buffers.z_buffers.axis_iter_mut(Axis(0)).collect();
    let mut image_views: Vec<_> = buffers.images.axis_iter_mut(Axis(0)).collect();
    let mut ocp_map_views: Vec<_> = buffers.ocp_maps.axis_iter_mut(Axis(0)).collect();

    z_buffer_views
        .par_iter_mut()
        .zip(image_views.par_iter_mut())
        .zip(ocp_map_views.par_iter_mut())
        .enumerate()
        .for_each(|(img_layer, ((z_slice, image_slice), ocp_slice))| {
            let projection_config = ProjectionConfig::new(img_layer);
            let num_faces = triangles.nrows();

            // Rasterize all triangles for this view
            for t_idx in 0..num_faces {
                rasterize_triangle(
                    t_idx,
                    points,
                    triangles,
                    triangle_uvs,
                    &triangle_normals, // Pass precomputed normals
                    &texture_data,
                    &projection_config,
                    z_slice,
                    image_slice,
                    ocp_slice,
                );
            }
        });
    conditional_log!(logging_enabled, info, "Rasterization finished");

    // --- Post-processing (Outlier Filtering) ---
    let outlier_counts = filter_outliers(
        &mut buffers,
        filtering_window_radius,
        rows,
        cols,
        max_bound_f32,
        channels,
        BACKGROUND_BRIGHTNESS,
        logging_enabled,
    );

    // --- Return Results ---
    (buffers.images, buffers.ocp_maps, outlier_counts)
}
