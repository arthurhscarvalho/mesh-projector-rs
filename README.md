# mesh-projector-rs

## Purpose

Rust libraries for loading textured or colored OBJ meshes, preprocessing geometry, rendering image projections, and sampling colored point clouds. The mesh projection and Triangular Area-Adaptive Point Sampling (TAPS) functionality supports the MAPPE mesh quality assessment pipeline.

## Related projects

[mesh-projector-pylib](https://github.com/arthurhscarvalho/mesh-projector-pylib) exposes these libraries to Python. [MAPPE](https://github.com/arthurhscarvalho/mappe) uses that interface to generate image and point-cloud inputs for quality assessment.

## Requirements

- A current stable Rust toolchain, Cargo, and a native linker/C compiler. The workspace uses Rust edition 2021.
- Network access to fetch Cargo dependencies.
- OBJ meshes with associated MTL and texture files where applicable.
- For GPU examples: a GPU adapter and drivers supported by the configured `wgpu` backends. On Linux, this normally requires a working Vulkan driver stack.

CPU rendering does not require a GPU adapter.

## Build

```bash
git clone https://github.com/arthurhscarvalho/mesh-projector-rs.git
cd mesh-projector-rs
cargo build --release --workspace
```

The workspace contains three library crates:

| Crate | Responsibility |
| --- | --- |
| `mesh-projector-core` | Mesh loading, geometry preprocessing, shared types, and rendering configuration. |
| `mesh-projector-cpu` | CPU image projection and surface sampling. |
| `mesh-projector-wgpu` | GPU image projection and surface sampling through `wgpu`. |

To build the CPU crates separately:

```bash
cargo build --release -p mesh-projector-core -p mesh-projector-cpu
```

## Usage

### Library workflow

Load the OBJ scene with `mesh_projector_core::mesh_io::read_mesh`, build a preprocessing recipe with `build_preprocessing_from_scene`, and apply it with `preprocessed_scene`. For a reference/distorted pair, derive the recipe from the reference and apply the same recipe to both scenes.

The CPU crate exposes `project_to_image` and `project_to_point_cloud`. The GPU crate exposes `WgpuProjector`; its image and point-cloud projection methods consume the same shared scene and configuration types. See the examples below for complete Rust calls.

### CPU projection example

Run from the repository root so that relative asset paths resolve:

```bash
mkdir -p projections
cargo run --release -p mesh-projector-cpu --example cpu_cube_projection
```

[crates/cpu/examples/cpu_cube_projection.rs](crates/cpu/examples/cpu_cube_projection.rs) loads the bundled [OBJ mesh](assets/just_a_girl_C0-L5_deq_tri.obj) and renders six views at resolution 4096. It saves `projection_*.png`, `depth_map_*.png`, `normal_map_*.png`, and `occupancy_map_*.png` in `projections/`. Edit `MESH_PATH` and `RESOLUTION` in the example to use another mesh or resolution.

### GPU projection examples

```bash
mkdir -p projections_wgpu
cargo run --release -p mesh-projector-wgpu --example wgpu_cube_projection
```

[crates/wgpu/examples/wgpu_cube_projection.rs](crates/wgpu/examples/wgpu_cube_projection.rs) uses the same bundled mesh and saves the corresponding maps in `projections_wgpu/`. Edit its `MESH_PATH` and `RESOLUTION` constants to change the input. View the available output-selection, MSAA, SSAO, and repeat options with:

```bash
cargo run --release -p mesh-projector-wgpu --example wgpu_cube_projection -- --help
```

Smaller examples exercise GPU rendering and point-cloud sampling at resolution 256 and print summaries without saving files:

```bash
cargo run --release -p mesh-projector-wgpu --example wgpu_image_smoke
cargo run --release -p mesh-projector-wgpu --example wgpu_point_cloud_smoke
```

These GPU examples require an available adapter; they return an error if initialization fails.

## Publications

The mesh processing and sampling methods are described as components of MAPPE:

1. Arthur Henrique Silva Carvalho and Pedro Garcia Freitas. “MAPPE: Mesh Assessment via Point Cloud and Projection Estimators.” *Signal Processing: Image Communication*, vol. 150, article 117722, 2027. [DOI: 10.1016/j.image.2026.117722](https://doi.org/10.1016/j.image.2026.117722). Available online 5 October 2026.

```bibtex
@article{carvalho2027mappe,
  author  = {Carvalho, Arthur Henrique Silva and Freitas, Pedro Garcia},
  title   = {{MAPPE}: Mesh Assessment via Point Cloud and Projection Estimators},
  journal = {Signal Processing: Image Communication},
  volume  = {150},
  pages   = {117722},
  year    = {2027},
  doi     = {10.1016/j.image.2026.117722}
}
```

## License

[MIT License](LICENSE).
