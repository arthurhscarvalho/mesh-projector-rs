# Mesh Projector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12.9-blue.svg)](https://www.python.org/)
[![Rust Version](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)

Generate 6 orthographic projections (+X, -X, +Y, -Y, +Z, -Z) from a 3D mesh, producing both color images and occlusion maps. This library implements a Rust backend via PyO3 for efficient computation, wrapped in a user-friendly Python interface.

## Features

* **Orthographic Projections:** Generates 6 projections along the major axes (front/back for X, Y, Z).
* **Multiple Rendering Modes:** The library automatically infers the rendering mode based on the provided files:
  * **OBJ Only:** Uses OBJ vertices and vertex colors (if available). Requires only OBJ file.
  * **OBJ + Texture:** Uses OBJ vertices/UVs and texture map. Requires OBJ and texture file.
  * **OBJ + Texture + MTL:** Uses OBJ vertices/UVs, texture map, and MTL properties. Requires OBJ, texture, and MTL files.
* **Mesh Loading:** Efficiently loads `.obj` mesh files directly in Rust.
* **UV Coordinate Processing:** Handles per-triangle UV coordinates, with configurable V-coordinate flipping.
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

### Python API

The library provides a single high-level function, `project_mesh()`, for generating projections. The repository includes an example script that demonstrates usage via command line:

```bash
# OBJ + Texture + MTL
python examples/example_generate_projections.py \
    --obj assets/just_a_girl_C0-L5_deq_tri.obj \
    --texture assets/just_a_girl_C0-L5_deq_tri_0.png \
    --mtl assets/just_a_girl_C0-L5_deq_tri.mtl \
    --out projections --precision 12 --verbose

# OBJ + Texture
python examples/example_generate_projections.py \
    --obj assets/just_a_girl_C0-L5_deq_tri.obj \
    --texture assets/just_a_girl_C0-L5_deq_tri_0.png \
    --out projections --verbose

# OBJ Only (requires vertex colors in OBJ or will be white)
python examples/example_generate_projections.py \
    --obj assets/Chameleon_Ref_0.obj \
    --out projections --verbose
```

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
