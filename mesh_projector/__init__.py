from textwrap import dedent
import os
import cv2
import numpy as np
import open3d as o3d
from loguru import logger

from .mesh_projector import (
    generate_projections as _internal_generate_projections,
)


def info(msg):
    logger.info(msg)


def _parse_mtl(mtl_path):
    """Parses relevant properties from an MTL file."""

    if not mtl_path or not os.path.exists(mtl_path):
        logger.warning(
            f"MTL file not provided or not found: {mtl_path}. Using default material values."
        )
        # Raw texture applied directly to the surface
        return {
            "ka": [0.0, 0.0, 0.0],  # Default Ambient
            "kd": [1.0, 1.0, 1.0],  # Default Diffuse
            "ks": [0.0, 0.0, 0.0],  # Default Specular
            "ns": 10.0,  # Default Shininess
            "opacity": 1.0,  # Default Opacity (from d or Tr)
            "illum": 1,  # Default Illumination model
        }

    material = {
        "ka": [0.1, 0.1, 0.1],  # Default Ambient
        "kd": [0.8, 0.8, 0.8],  # Default Diffuse
        "ks": [0.1, 0.1, 0.1],  # Default Specular
        "ns": 10.0,  # Default Shininess
        "opacity": 1.0,  # Default Opacity (from d or Tr)
        "illum": 2,  # Default Illumination model
    }

    try:
        with open(mtl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                key = parts[0].lower()
                values = parts[1:]

                if key == "ka" and len(values) == 3:
                    material["ka"] = [float(v) for v in values]
                elif key == "kd" and len(values) == 3:
                    material["kd"] = [float(v) for v in values]
                elif key == "ks" and len(values) == 3:
                    material["ks"] = [float(v) for v in values]
                elif key == "ns" and len(values) == 1:
                    material["ns"] = float(values[0])
                elif key == "tr" and len(values) == 1:  # Transparency (1 - opacity)
                    material["opacity"] = 1.0 - float(values[0])
                elif key == "d" and len(values) == 1:  # Dissolve (opacity)
                    material["opacity"] = float(values[0])
                elif key == "illum" and len(values) == 1:
                    material["illum"] = int(values[0])
                # map_Kd is ignored here as the texture data is passed separately

        # Clamp opacity
        material["opacity"] = max(0.0, min(1.0, material["opacity"]))
        info(f"Parsed material properties from {mtl_path}: {material}")

    except Exception as e:
        logger.error(f"Error parsing MTL file {mtl_path}: {e}. Using default values.")
        # Raw texture applied directly to the surface
        return {
            "ka": [0.0, 0.0, 0.0],  # Default Ambient
            "kd": [1.0, 1.0, 1.0],  # Default Diffuse
            "ks": [0.0, 0.0, 0.0],  # Default Specular
            "ns": 10.0,  # Default Shininess
            "opacity": 1.0,  # Default Opacity (from d or Tr)
            "illum": 1,  # Default Illumination model
        }

    return material


def read_mesh(
    obj_location,
    texture_file,
    mtl_location=None,
    enable_post_processing=False,
    verbose=False,
):
    """
    Loads a mesh and its texture (external).

    Args:
        obj_location (str): Path to the OBJ mesh file.
        texture_file (str): Path to the texture image file.
        mtl_location (str, optional): Path to the MTL material file. Defaults to None.
        enable_post_processing (bool): Enables Open3D post-processing (e.g., vertex merging).
        verbose (bool): Enables detailed logging output.

    Returns:
        o3d.geometry.TriangleMesh: The loaded mesh with texture attached.

    Raises:
        ValueError, FileNotFoundError: On failure to load or missing components.
    """
    if verbose is not True:
        logger.remove()
    info(f"Loading mesh: {obj_location}")
    info(f"Post-processing enabled: {enable_post_processing}")
    mesh = o3d.io.read_triangle_mesh(
        obj_location,
        enable_post_processing=enable_post_processing,
    )
    if not mesh.has_vertices():
        raise ValueError(f"Mesh '{obj_location}' has no vertices!")
    if not mesh.has_triangle_uvs():
        err = f"Mesh '{obj_location}' has no UV coordinates, cannot use texture."
        raise ValueError(err)
    texture = cv2.cvtColor(cv2.imread(texture_file), cv2.COLOR_BGR2RGB)
    mesh.textures = [o3d.geometry.Image(texture)]
    info(f"Texture load successful: {texture_file}")
    # Note: We load the MTL separately if needed, Open3D doesn't directly apply it here
    # in a way that's easily transferable to our custom renderer.
    return mesh


def preprocess_mesh(mesh, precision, flip_v_coordinate=False, verbose=False):
    """
    Processes a mesh by scaling vertices and adjusting UVs for use with texture.

    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh with texture and UVs.
        precision (int): A power-of-2 exponent determining the output image resolution.
                         The output images will be `(2^precision) x (2^precision)`.
        flip_v_coordinate (bool): Whether to flip the V coordinate (1.0 - v).
        verbose (bool): Enables detailed logging.

    Returns:
        tuple:
            - vertices (np.ndarray[float32]): Scaled vertex positions (N, 3).
            - triangle_indices (np.ndarray[uint64]): Triangle vertex indices (M, 3).
            - texture_np (np.ndarray[uint8]): Texture image data (H, W, C).
            - triangle_uvs (np.ndarray[float32]): Per-triangle UV coordinates (M*3, 2).
    """
    if verbose is not True:
        logger.remove()
    texture_np = np.asarray(mesh.textures[0])
    if texture_np.ndim != 3 or texture_np.shape[2] < 3:
        err = f"Texture has shape {texture_np.shape}. Expected 3 channels (RGB)."
        raise ValueError(err)
    tex_height, tex_width, tex_channels = texture_np.shape
    info(f"Texture shape: H={tex_height}, W={tex_width}, C={tex_channels}")
    triangle_uvs = np.asarray(mesh.triangle_uvs)
    if triangle_uvs.shape[1] != 2:
        err = f"Expected triangle_uvs shape to be (*, 2), but got {triangle_uvs.shape}"
        raise ValueError(err)
    # Clamp UVs to [0, 1] range to avoid issues with sampling outside bounds
    triangle_uvs = np.clip(triangle_uvs, 0.0, 1.0)
    # IMPORTANT: Optionally flip the V (vertical) coordinate.
    if flip_v_coordinate:
        info("Flipping V coordinate (1.0 - v)")
        triangle_uvs[:, 1] = 1.0 - triangle_uvs[:, 1]
    else:
        info("Using original V coordinates")
    # --- Vertex Data ---
    vertices = np.asarray(mesh.vertices)
    triangle_indices = np.asarray(mesh.triangles, dtype=np.uint32)
    if triangle_indices.size == 0:
        raise ValueError("Mesh has vertices but no triangles!")
    if triangle_uvs.shape[0] != triangle_indices.size:
        err = f"UV mismatch: UVs ({triangle_uvs.shape[0]}) vs. triangles ({triangle_indices.size})"
        raise ValueError(err)
    # --- Vertex Scaling ---
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)
    info(f"Original BBox Min: {min_bound}, Max: {max_bound}")
    # Shift if necessary to make min bound >= 0
    shift = np.maximum(0, -min_bound)
    if np.any(shift > 0):
        info(f"Shifting vertices by: {shift}")
        vertices += shift
        min_bound = vertices.min(axis=0)
        max_bound = vertices.max(axis=0)
        info(f"Shifted BBox Min: {min_bound}, Max: {max_bound}")
    bbox_size = max_bound - min_bound
    # Handle zero size dimensions if any (e.g., flat mesh)
    bbox_size[bbox_size == 0] = 1e-6  # Avoid division by zero, set a tiny size
    # Uniform scaling based on the *largest* dimension
    max_dim = np.max(bbox_size)
    # Target maximum coordinate value (e.g., 65535.0 for precision=16)
    target_max_val = (1 << precision) - 1.0
    scale_factor = target_max_val / max_dim
    info(f"Scaling vertices. Max dimension: {max_dim}, Scale factor: {scale_factor}")
    vertices *= scale_factor
    # Update bounds after scaling
    min_bound_scaled = vertices.min(axis=0)
    max_bound_scaled = vertices.max(axis=0)
    info(f"Scaled BBox Min: {min_bound_scaled}, Max: {max_bound_scaled}")
    final_msg = dedent(f"""
        Final Data Shapes:
            Vertices (scaled): {vertices.shape}, dtype={vertices.dtype}
            Triangle Indices: {triangle_indices.shape}, dtype={triangle_indices.dtype}
            Texture: {texture_np.shape}, dtype={texture_np.dtype}
            Triangle UVs (processed): {triangle_uvs.shape}, dtype={triangle_uvs.dtype}
    """).strip()
    info(final_msg)
    # Return data needed by Rust
    return (
        vertices.astype(np.float32),
        triangle_indices.astype(np.uint64),
        texture_np.astype(np.uint8),
        triangle_uvs.astype(np.float32),
    )


def generate_projections(
    mesh,
    precision,
    mtl_location=None,
    filtering=0,
    verbose=False,
):
    """
    Computes 6 orthographic projections (front/back for X/Y/Z) of a textured mesh.

    This function calls a Rust back-end to perform the projection calculations.

    Args:
        mesh: (o3d.geometry.TriangleMesh): An open3d mesh with texture attached.
        precision (int): A power-of-2 exponent determining the output image resolution.
                         The output images will be `(2^precision) x (2^precision)`.
        mtl_location (str, optional): Path to the MTL file for material properties. Defaults to None.
        filtering (int, optional): Radius of the square neighborhood window for outlier filtering. Defaults to 0.
                                   A value of 0 disables the filtering step.
        verbose (bool, optional): Flag to enable or disable logging messages. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - images (np.ndarray[np.uint8]): An array of shape (6, H, W, C) holding the 6 rendered projection images.
            - ocp_maps (np.ndarray[np.float32]): An array of shape (6, H, W) holding the 6 occlusion maps,
                                                where 1.0 indicates a mesh pixel and 0.0 indicates background.
    """
    if verbose is not True:
        logger.remove()

    info("Parsing MTL file (if provided)")
    material_props = _parse_mtl(mtl_location)

    info("Preprocessing mesh")
    mesh_data = preprocess_mesh(mesh, precision, verbose=verbose)
    info("Sending data to Rust back-end")

    # Unpack mesh data
    vertices, triangle_indices, texture_np, triangle_uvs = mesh_data

    # Call the Rust backend with material properties
    images, ocp_maps = _internal_generate_projections(
        vertices,
        triangle_indices,
        texture_np,
        triangle_uvs,
        precision,
        material_props["ka"],  # Ambient color (list/tuple)
        material_props["kd"],  # Diffuse color (list/tuple)
        material_props["ks"],  # Specular color (list/tuple)
        material_props["ns"],  # Shininess (float)
        material_props["opacity"],  # Opacity (float)
        material_props["illum"],  # Illumination model (int)
        filtering,
        verbose,
    )
    images, ocp_maps = np.asarray(images), np.asarray(ocp_maps)
    info("Projections successfully generated!")
    return images, ocp_maps
