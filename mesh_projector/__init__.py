from .mesh_projector import (
    generate_projections_rotation as _internal_generate_projections,
)
from .mesh_projector import generate_point_cloud as _internal_generate_point_cloud
from .mesh_projector import (
    generate_cube_projections as _internal_generate_cube_projections,
)


def generate_point_cloud(
    obj_path,
    texture_path=None,
    mtl_path=None,
    precision=12,
    filtering_radius=0,
    flip_uv=True,
    point_cloud_density_factor=1.0,
    ignore_opacity=False,
    verbose=False,
):
    return _internal_generate_point_cloud(
        obj_path,
        texture_path,
        mtl_path,
        precision,
        filtering_radius,
        flip_uv,
        point_cloud_density_factor,
        ignore_opacity,
        verbose,
    )


def project_mesh(
    obj_location,
    texture_file=None,
    mtl_location=None,
    precision=12,
    filtering=0,
    flip_uv=True,
    ignore_opacity=False,
    verbose=False,
    rotation_steps=None,
    angle_delta_x=None,
    angle_delta_y=None,
    angle_delta_z=None,
):
    """
    High-level function to load and generate projections and occlusion maps for a mesh.

    Args:
        obj_location (str): Path to the OBJ mesh file.
        precision (int): Output image resolution exponent (2^precision).
        texture_file (str, optional): Path to the texture image file. Defaults to None.
        mtl_location (str, optional): Path to the MTL material file. Defaults to None.
        filtering (int, optional): Radius for outlier filtering. If > 0, occlusion maps are generated.
                                 Defaults to 0 (disabled).
        flip_uv (bool): Whether to flip the V coordinate (1.0 - v) during preprocessing.
                      Defaults to True.
        verbose (bool): Enables detailed logging. Defaults to False.
        ignore_opacity (bool): If True, forces opacity to 1.0, ignoring the MTL value. Defaults to False.
        rotation_steps (int, optional): Number of rotation steps for rotation-based projection.
                                     If provided along with angle deltas, enables rotation-based projection.
                                     Defaults to None (uses standard 6-view orthographic projection).
        angle_delta_x (float, optional): Angle delta for rotation around X axis in radians.
                                      Required for rotation-based projection.
        angle_delta_y (float, optional): Angle delta for rotation around Y axis in radians.
                                      Required for rotation-based projection.
        angle_delta_z (float, optional): Angle delta for rotation around Z axis in radians.
                                      Required for rotation-based projection.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - Projection images (N, H, W, 3) where N is either 6 (orthographic) or rotation_steps (rotation-based)
            - Depth maps (N, H, W, 3)
            - Normal maps (N, H, W, 3)
            - Occupancy maps (N, H, W) where value is 255 for occupied pixels, 0 otherwise

    Raises:
        FileNotFoundError: If required files (OBJ, or Texture/MTL based on mode) are not found.
        ValueError: If the mesh data is invalid or required components are missing.
        RuntimeError: If the Rust backend fails during projection generation.
    """
    try:
        # The mode is automatically inferred in the Rust backend based on provided files
        _internal_generate_projections(
            obj_location,
            texture_file,
            mtl_location,
            precision,
            filtering,
            flip_uv,
            ignore_opacity,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")


def project_mesh_cube(
    obj_location,
    texture_file=None,
    mtl_location=None,
    precision=12,
    filtering=0,
    flip_uv=True,
    ignore_opacity=False,
    verbose=False,
):
    try:
        # The mode is automatically inferred in the Rust backend based on provided files
        return _internal_generate_cube_projections(
            obj_location,
            texture_file,
            mtl_location,
            precision,
            filtering,
            flip_uv,
            ignore_opacity,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")
