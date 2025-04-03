# Mesh Projector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12.9-blue.svg)](https://www.python.org/)
[![Rust Version](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

Generate 6 orthographic projections (+X, -X, +Y, -Y, +Z, -Z) from a textured 3D mesh, producing both color images and occlusion maps. This library implements a Rust backend via PyO3 for efficient computation, wrapped in a user-friendly Python interface.

## Features

* **Orthographic Projections:** Generates 6 projections along the major axes (front/back for X, Y, Z).
* **Textured Mesh Handling:** Processes meshes with associated texture maps (requires separate texture file).
* **Mesh Loading:** Utilizes Open3D for loading `.obj` mesh files.
* **UV Coordinate Processing:** Handles per-triangle UV coordinates, including an option to flip the V coordinate if needed.
* **Vertex Scaling:** Automatically scales and shifts mesh vertices to fit within a specified precision range for consistent projection dimensions.
* **Performance:** Core rasterization and projection logic implemented in Rust using `ndarray` and `rayon` for parallelism, providing significant speed advantages.
* **Outlier Filtering:** Optional depth-based outlier filtering to remove isolated pixels far from their neighbours.
* **Outputs:** Provides both the rendered RGB projection images and floating-point occlusion maps (1.0 for mesh, 0.0 for background).

## Installation

### Prerequisites

1.  **Rust Toolchain:** Required to build the Rust backend. Install from [rustup.rs](https://rustup.rs/).
2.  **Conda (Recommended):** For managing Python environments and dependencies. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).
3.  **Python:** Version 3.12.9 is specified.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/arthurhscarvalho/mesh-projector
    cd mesh-projector
    ```

2.  **Create and activate the Conda environment:**
    This will install Python, Maturin (the Rust-Python build tool), and necessary Python packages.
    ```bash
    conda env create -f environment.yml
    conda activate mesh_projector
    ```

3.  **Build and install the package:**
    Maturin will compile the Rust code and install the Python package in your active environment.
    ```bash
    maturin develop -r
    ```

## Usage

The following example demonstrates how to load a mesh and generate projections:

```python
import os
import time
import cv2
import numpy as np
import mesh_projector # Import the library

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
    # e.g., precision=8 -> 256x256 images
    precision = 12 # Results in 4096x4096 images
    # Outlier filtering radius (0 disables filtering)
    filtering = 0
    verbose = True # Enable logging messages

    # --- Load Mesh ---
    # Provide paths to your mesh and texture files
    obj_path = "assets/just_a_girl_C0-L5_deq_tri.obj"
    tex_path = "assets/just_a_girl_C0-L5_deq_tri_0.png" # Or associated texture like .jpg

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
        # Optional: flip V texture coordinate if needed (e.g., Blender exports)
        # flip_v_coordinate=False,
        verbose=verbose
    )
    print("Mesh loaded successfully.")

    # --- Generate Projections ---
    print("Generating projections...")
    t0 = time.time()
    # Call the core function (which calls the Rust backend)
    images, ocp_maps = mesh_projector.generate_projections(
        *mesh_data, # Unpack the tuple from read_mesh
        precision,
        filtering,
        verbose=verbose
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
```

## API Overview

* **`mesh_projector.read_mesh(obj_location, texture_file, precision, enable_post_processing=False, flip_v_coordinate=False, verbose=False)`**
    * Loads the mesh (`.obj`) and its texture using Open3D.
    * Processes geometry and UVs.
    * Scales vertices uniformly based on `precision` to fit the range `[0, 2^precision - 1]`.
    * Handles potential texture loading issues and missing UVs.
    * Optionally flips the V texture coordinate (`flip_v_coordinate=True`).
    * Returns a tuple containing NumPy arrays: `(vertices, triangle_indices, texture_np, triangle_uvs)` formatted for the projection function.

* **`mesh_projector.generate_projections(points, triangles, texture, triangle_uvs, precision, filtering=0, verbose=False)`**
    * Takes the processed mesh data (output of `read_mesh`).
    * Calls the high-performance Rust backend to perform rasterization and projection.
    * `precision`: Determines output resolution (`2^precision x 2^precision`).
    * `filtering`: Radius for outlier removal (0 to disable).
    * Returns a tuple: `(images, ocp_maps)`
        * `images`: NumPy array `(6, H, W, C)` of type `uint8` containing the 6 RGB projection images.
        * `ocp_maps`: NumPy array `(6, H, W)` of type `float32` containing the occlusion maps (1.0 for mesh pixels, 0.0 for background).

## Dependencies

### Python

* Python == 3.12.9
* numpy >= 2.2
* open3d == 0.19.0
* toml >= 0.10.2
* loguru >= 0.7
* opencv-python (optional, for saving images in example)

### Build/Runtime

* Rust toolchain (stable, 2021 edition)
* maturin >= 1.0, < 2.0
* patchelf (Linux only, via conda)

### Rust Crates (Managed by Cargo/Maturin)

* pyo3 (for Python bindings)
* numpy (Rust crate for interacting with NumPy)
* ndarray (for numerical operations)
* rayon (for parallelization)
* log, colog (for logging)

## Development

1.  Set up the environment as described in the **Installation** section.
2.  Use `maturin develop -r` to build and install the package after making changes to Rust (`src/`) or Python (`mesh_projector/`) code.
3.  Use the `format_code.sh` script to format and lint Python code using `ruff`:
    ```bash
    bash format_code.sh
    ```
4.  Format Rust code using `cargo fmt`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request on the [GitHub repository](https://github.com/arthurhscarvalho/mesh-projector).
