import numpy as np

from .mesh_projector import (
    generate_projections as _internal_generate_projections,
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

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]:
            - A list of 6 numpy arrays, each representing a projection image (H, W, 3).
            - A list of 6 numpy arrays, each representing an outlier occlusion map (H, W).
              Value is 1 for filtered outlier pixels, 0 otherwise. Returns empty maps if filtering=0.

    Raises:
        FileNotFoundError: If required files (OBJ, or Texture/MTL based on mode) are not found.
        ValueError: If the mesh data is invalid or required components are missing.
        RuntimeError: If the Rust backend fails during projection generation.
    """
    try:
        # The mode is automatically inferred in the Rust backend based on provided files
        projection_array, occlusion_array = _internal_generate_projections(
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
    # Convert the 4D arrays to lists of images/maps
    # Projection Images (6, H, W, 3)
    projection_images = [projection_array[i] for i in range(projection_array.shape[0])]
    # Occlusion Maps (6, H, W, 1) -> need to squeeze to (6, H, W)
    occlusion_maps = [
        np.squeeze(occlusion_array[i], axis=-1) for i in range(occlusion_array.shape[0])
    ]
    return projection_images, occlusion_maps
