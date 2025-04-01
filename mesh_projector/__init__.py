import os

import numpy as np
import open3d as o3d
from loguru import logger

from textwrap import dedent

from .mesh_projector import (
    generate_projections as _internal_generate_projections,
)


def read_mesh(
    obj_location,
    texture_file,
    precision,
    enable_post_processing=False,  # Default to False for safer texture mapping
    flip_v_coordinate=True,  # Default to True as it's common, but make it configurable
    verbose=False,
):
    """
    Reads a mesh and its texture, processes geometry and UVs, preparing data for export.

    Args:
        obj_location (str): Path to the OBJ mesh file.
        texture_file (str): Path to the texture image file.
        precision (int): The bit precision for vertex scaling (e.g., 16 for 16-bit).
        enable_post_processing (bool): If True, enables Open3D's post-processing
                                       (can merge vertices, potentially breaking UV seams).
                                       Defaults to False.
        flip_v_coordinate (bool): If True, flips the V texture coordinate (1.0 - v).
                                  Needed if texture origin is top-left and UVs assume
                                  bottom-left. Defaults to True.

    Returns:
        tuple: Contains:
            - vertices (np.ndarray[np.float64]): Scaled vertex positions (N, 3).
            - triangle_indices (np.ndarray[np.uint32]): Triangle vertex indices (M, 3).
                                                        Using uint32 as often sufficient.
            - texture_np (np.ndarray[np.uint8]): Texture image data (H, W, C).
            - triangle_uvs (np.ndarray[np.float64]): Per-triangle-corner UV coords (M*3, 2).

    Raises:
        ValueError: If the mesh or texture cannot be loaded or lacks required components.
        FileNotFoundError: If the texture file does not exist when explicit loading is attempted.
    """
    if verbose is not True:
        logger.remove()
    logger.info(f"Loading mesh: {obj_location}")
    logger.info(f"Post-processing enabled: {enable_post_processing}")
    logger.info(f"Flip V coordinate: {flip_v_coordinate}")

    mesh = o3d.io.read_triangle_mesh(
        obj_location, enable_post_processing=enable_post_processing
    )

    if not mesh.has_vertices():
        raise ValueError(f"Mesh '{obj_location}' has no vertices!")

    # --- Texture Handling ---
    texture_loaded_explicitly = False
    # Check if textures are already loaded (e.g., from MTL or embedded)
    if not mesh.has_textures() or not mesh.textures or mesh.textures[0].is_empty():
        logger.info(
            f"Mesh '{obj_location}' has no embedded/MTL texture or it's invalid. Attempting explicit load..."
        )
        if not os.path.exists(texture_file):
            raise FileNotFoundError(f"Texture file not found: {texture_file}")
        try:
            texture_img = o3d.io.read_image(texture_file)
            if texture_img.is_empty():
                raise ValueError("Loaded texture image is empty.")
            mesh.textures = [texture_img]
            texture_loaded_explicitly = True
            logger.info(f"Successfully loaded texture explicitly: {texture_file}")
        except Exception as e:
            raise ValueError(
                f"Mesh has no texture and failed to load explicitly from {texture_file}: {e}"
            )
    elif not mesh.textures[0].is_empty():
        logger.info("Texture found associated with the mesh (e.g., via MTL file).")
    else:
        # This case should ideally not be hit due to the outer check, but belt-and-suspenders
        raise ValueError(
            f"Mesh texture list was present but the first texture is empty/invalid. Path: {obj_location}"
        )

    # --- UV Coordinate Handling ---
    if not mesh.has_triangle_uvs():
        # If we loaded texture explicitly, maybe the mesh *never* had UVs
        if texture_loaded_explicitly:
            raise ValueError(
                f"Mesh '{obj_location}' has no UV coordinates, cannot use texture '{texture_file}'."
            )
        else:
            # If texture came with mesh (MTL), lack of UVs is strange
            raise ValueError(
                f"Mesh '{obj_location}' loaded with texture data but lacks UV coordinates!"
            )

    # Use the first texture associated with the mesh
    texture_o3d = mesh.textures[0]
    texture_np = np.asarray(
        texture_o3d
    )  # Convert to NumPy array (H, W, C) usually uint8
    if texture_np.ndim != 3 or texture_np.shape[2] < 3:
        # Handle grayscale images - potentially convert them to RGB if needed by Rust
        # Or raise error if color is expected. For now, just warn.
        logger.warning(
            f"Texture has shape {texture_np.shape}. Expected 3 channels (RGB)."
        )
        # Example: Convert grayscale to RGB if necessary downstream
        # if texture_np.ndim == 2:
        #     texture_np = np.stack((texture_np,) * 3, axis=-1)
        # elif texture_np.shape[2] == 1:
        #      texture_np = np.concatenate((texture_np,) * 3, axis=2)

    tex_height, tex_width, tex_channels = texture_np.shape
    logger.info(f"Texture shape: H={tex_height}, W={tex_width}, C={tex_channels}")

    # Get per-triangle UVs. Open3D stores them flattened: (t1_v0_u, t1_v0_v, t1_v1_u, t1_v1_v, ...)
    # Reshape them to (num_triangles * 3, 2) for easier processing
    # Note: Open3D 0.15+ returns (N, 2) directly. Ensure your version does.
    # If using older Open3D, you might need:
    # uvs_flat = np.asarray(mesh.triangle_uvs)
    # triangle_uvs = uvs_flat.reshape(-1, 2)
    # Assuming modern Open3D:
    triangle_uvs = np.asarray(mesh.triangle_uvs)  # Shape: (num_triangles * 3, 2)

    if triangle_uvs.shape[1] != 2:
        raise ValueError(
            f"Expected triangle_uvs shape to be (*, 2), but got {triangle_uvs.shape}"
        )

    logger.info(f"Raw triangle UVs shape: {triangle_uvs.shape}")
    # Clamp UVs to [0, 1] range to avoid issues with sampling outside bounds
    triangle_uvs = np.clip(triangle_uvs, 0.0, 1.0)

    # IMPORTANT: Optionally flip the V (vertical) coordinate.
    if flip_v_coordinate:
        logger.info("Flipping V coordinate (1.0 - v)")
        triangle_uvs[:, 1] = 1.0 - triangle_uvs[:, 1]
    else:
        logger.info("Using original V coordinate")

    # --- Vertex Data ---
    vertices = np.asarray(mesh.vertices)  # XYZ positions
    # Using uint32 for indices - adjust to uint64 if necessary for huge meshes
    triangle_indices = np.asarray(mesh.triangles, dtype=np.uint32)

    logger.info(f"Original vertices shape: {vertices.shape}")
    logger.info(f"Triangle indices shape: {triangle_indices.shape}")

    if triangle_indices.size == 0:
        raise ValueError("Mesh has vertices but no triangles!")
    if triangle_uvs.shape[0] != triangle_indices.size:
        raise ValueError(
            f"Mismatch between number of UV coordinates ({triangle_uvs.shape[0]}) "
            f"and number of triangle corners ({triangle_indices.size}). "
            f"Check mesh integrity or OBJ exporter settings."
        )

    # --- Vertex Scaling (Keep this logic) ---
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    logger.info(f"Original BBox Min: {min_bound}, Max: {max_bound}")

    # Shift if necessary to make min bound >= 0
    shift = np.maximum(0, -min_bound)  # Calculate shift only for negative components
    if np.any(shift > 0):
        logger.info(f"Shifting vertices by: {shift}")
        vertices += shift
        # Recalculate bounds after shift
        min_bound = vertices.min(axis=0)  # Should be >= 0 now
        max_bound = vertices.max(axis=0)
        logger.info(f"Shifted BBox Min: {min_bound}, Max: {max_bound}")

    bbox_size = max_bound - min_bound  # Size along each axis

    # Handle zero size dimensions if any (e.g., flat mesh)
    bbox_size[bbox_size == 0] = 1e-6  # Avoid division by zero, set a tiny size

    # Uniform scaling based on the *largest* dimension
    max_dim = np.max(bbox_size)
    # Target maximum coordinate value (e.g., 65535.0 for precision=16)
    target_max_val = (1 << precision) - 1.0
    scale_factor = target_max_val / max_dim
    # Apply scaling slightly less than full range to potentially avoid edge cases if clamping downstream
    # scale_factor *= (1.0 - 1e-6 / (1 << precision)) # Optional: slightly smaller scale

    logger.info(
        f"Scaling vertices. Max dimension: {max_dim}, Scale factor: {scale_factor}"
    )
    vertices *= scale_factor

    # Update bounds after scaling
    min_bound_scaled = vertices.min(axis=0)
    max_bound_scaled = vertices.max(axis=0)
    logger.info(f"Scaled BBox Min: {min_bound_scaled}, Max: {max_bound_scaled}")

    # Optional: Clamp vertices to ensure they are within [0, target_max_val]
    # This can be useful if floating point inaccuracies push values slightly out
    # vertices = np.clip(vertices, 0.0, target_max_val)
    # print("Clamped scaled vertices to [0, target_max_val]")

    final_info = dedent(f"""
        Final Data Shapes:
            Vertices (scaled): {vertices.shape}, dtype={vertices.dtype}
            Triangle Indices: {triangle_indices.shape}, dtype={triangle_indices.dtype}
            Texture: {texture_np.shape}, dtype={texture_np.dtype}
            Triangle UVs (processed): {triangle_uvs.shape}, dtype={triangle_uvs.dtype}
    """).strip()
    logger.info(final_info)

    # Return data needed by Rust
    return (
        vertices.astype(np.float64),
        triangle_indices.astype(np.uint64),
        texture_np.astype(np.uint8),
        triangle_uvs.astype(np.float64),
    )


def generate_projections(
    points,
    triangles,
    texture,
    triangle_uvs,
    precision,
    filtering=2,
    verbose=False,
):
    if verbose is not True:
        logger.remove()
    logger.info("Sending data to Rust back-end")
    images, ocp_maps = _internal_generate_projections(
        points, triangles, texture, triangle_uvs, precision, filtering, verbose
    )
    images, ocp_maps = np.asarray(images), np.asarray(ocp_maps)
    logger.info("Projections successfully generated!")
    return images, ocp_maps
