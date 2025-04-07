import os
import time

import cv2
import numpy as np

import mesh_projector  # Import the library


def save_projections(projections, output_dir="./projections"):
    """Helper function to save projection images."""
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(projections)):
        image = projections[i].astype(np.uint8)
        # Convert RGB (from library) to BGR (for OpenCV)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"projection_{i}.png"), image_bgr)
    print(f"Projections saved to {output_dir}")


if __name__ == "__main__":
    # --- Configuration ---
    # Precision determines output resolution: 2^precision x 2^precision
    # e.g., precision=10 -> 1024x1024 images
    precision = 12  # Results in 4096x4096 images
    filtering = 0  # Outlier filtering radius (0 disables filtering)
    verbose = True  # Enable logging messages

    # --- Load Mesh ---
    # Provide paths to your mesh and texture files
    obj_path = "assets/just_a_girl_C0-L5_deq_tri.obj"
    tex_path = "assets/just_a_girl_C0-L5_deq_tri_0.png"

    # read_mesh prepares data for the Rust backend
    # It handles loading, vertex scaling, and UV processing.
    # Returns a tuple: (vertices, triangle_indices, texture_np, triangle_uvs)
    print("Loading mesh...")
    mesh_data = mesh_projector.read_mesh(
        obj_path,
        tex_path,
        precision,
        # Optional: enable Open3D post-processing (can break UVs)
        # enable_post_processing=False,
        # Optional: flip V texture coordinate if needed
        # flip_v_coordinate=False,
        verbose=verbose,
    )
    print("Mesh loaded successfully.")

    # --- Generate Projections ---
    print("Generating projections...")
    t0 = time.time()
    # Call the core function (which calls the Rust backend)
    images, ocp_maps = mesh_projector.generate_projections(
        mesh_data, precision, filtering, verbose
    )
    t1 = time.time()
    print(f"Projections generated. Time taken: {(t1 - t0):.2f} s")

    # --- Process Outputs ---
    # `images` is a NumPy array: (6, H, W, C) uint8
    # `ocp_maps` is a NumPy array: (6, H, W) float32 (1.0=mesh, 0.0=background)
    print(f"Output images shape: {images.shape}")
    print(f"Output occlusion maps shape: {ocp_maps.shape}")

    # Save the generated images
    save_projections(images)

    # You can now use `images` and `ocp_maps` for further processing.
