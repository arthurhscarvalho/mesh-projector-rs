from typing import Optional, Tuple

import numpy as np

from .mesh_projector import MeshWrapper
from .mesh_projector import (
    project_to_image_by_rotation as _internal_project_to_image_by_rotation,
)
from .mesh_projector import (
    project_to_image_by_rotation_to_file as _internal_project_to_image_by_rotation_to_file,
)
from .mesh_projector import (
    project_to_image_on_bounding_cube as _internal_project_to_image_on_bounding_cube,
)
from .mesh_projector import (
    project_to_point_cloud_by_sampling as _internal_project_to_point_cloud_by_sampling,
)
from .mesh_projector import (
    read_mesh as _internal_read_mesh,
)


class Mesh:
    """
    Represents a mesh object for use with the mesh projector.

    This class serves as a Pythonic wrapper around the internal mesh representation used by the Rust backend.
    It is intended to provide a convenient interface for mesh operations within the mesh projector package.
    Users should not instantiate this class directly; instead, obtain instances via the `read_mesh` function.
    This class is intended for internal use within the mesh projector and is not designed for use outside this context.
    """

    def __init__(self, mesh: MeshWrapper):
        self._internal_mesh = mesh


def read_mesh(
    obj_path: str,
    texture_path: Optional[str] = None,
    mtl_path: Optional[str] = None,
    precision: int = 12,
    flip_uv: bool = True,
    ignore_opacity: bool = False,
    verbose: bool = False,
) -> Mesh:
    """
    Load a 3D mesh from disk, automatically inferring the optimal loading mode based on the provided files.

    This function supports three loading modes, selected automatically:
        - OBJ Only: Loads mesh geometry and vertex colors (if present) from the OBJ file.
        - OBJ + Texture: Loads mesh geometry and UVs from the OBJ file, and applies the provided texture image.
        - OBJ + Texture + MTL: Loads mesh geometry, UVs, and material properties from the OBJ and MTL files, and applies the texture image.

    The mesh is loaded into a format compatible with the Rust backend for efficient processing.

    Args:
        obj_path (str): Path to the OBJ file containing mesh geometry.
        texture_path (Optional[str], optional): Path to the texture image file (if available).
        mtl_path (Optional[str], optional): Path to the MTL material file (if available).
        precision (int, optional): Precision for mesh normalization and rasterization (default: 12).
            Higher precision results in higher resolution.
            The mesh will be scaled to 2^precision.
            The default precision is 12, which results in 4096x4096 resolution for the bounding cube.
        flip_uv (bool, optional): Whether to flip the V coordinate of UVs (default: True).
        ignore_opacity (bool, optional): If True, ignore opacity/alpha in materials and textures (default: False).
        verbose (bool, optional): If True, enable verbose logging from the Rust backend (default: False).

    Returns:
        Mesh: A Mesh object ready for projection and point cloud operations.

    Raises:
        RuntimeError: If the mesh could not be loaded by the Rust backend.
    """
    try:
        internal_mesh = _internal_read_mesh(
            obj_path,
            None,
            precision,
            flip_uv,
            ignore_opacity,
            verbose,
        )
        return Mesh(internal_mesh)
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during mesh reading: {e}")


def project_to_image_on_bounding_cube(
    mesh: Mesh,
    filtering: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 6 orthographic projections of the mesh, one for each face of the bounding cube.

    Args:
        mesh (Mesh): The mesh object to project (as returned by `read_mesh`).
        filtering (int, optional): Radius for outlier filtering in the depth map (default: 0, disables filtering).
        verbose (bool, optional): If True, enables verbose logging from the Rust backend (default: False).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - projections: (6, H, W, 3) uint8 RGB images for each cube face
            - depth_maps: (6, H, W) uint8 depth maps for each face
            - normal_maps: (6, H, W, 3) uint8 normal maps for each face
            - occlusion_maps: (6, H, W) uint8 occupancy maps for each face (255=mesh, 0=background)

    Raises:
        RuntimeError: If the Rust backend fails during projection generation.
    """
    try:
        return _internal_project_to_image_on_bounding_cube(
            mesh._internal_mesh,
            filtering,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")


def project_to_image_by_rotation(
    mesh: Mesh,
    rotation_steps: int,
    angle_delta_x: float,
    angle_delta_y: float,
    angle_delta_z: float,
    filtering: int = 0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a batch of orthographic projections of the mesh by rotating it in 3D space.

    Args:
        mesh (Mesh): The mesh object to project (as returned by `read_mesh`).
        rotation_steps (int): Number of rotation steps to perform. The mesh will be rotated this many times, generating a projection at each step.
        angle_delta_x (float): Rotation angle increment (in radians) around the X axis for each step.
        angle_delta_y (float): Rotation angle increment (in radians) around the Y axis for each step.
        angle_delta_z (float): Rotation angle increment (in radians) around the Z axis for each step.
        filtering (int, optional): Radius for outlier filtering in the depth map (default: 0, disables filtering).
        verbose (bool, optional): If True, enables verbose logging from the Rust backend (default: False).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - projections: (N, H, W, 3) uint8 RGB images for each rotation step
            - depth_maps: (N, H, W) uint8 depth maps for each rotation step
            - normal_maps: (N, H, W, 3) uint8 normal maps for each rotation step
            - occlusion_maps: (N, H, W) uint8 occupancy maps for each rotation step (255=mesh, 0=background)

    Raises:
        RuntimeError: If the Rust backend fails during projection generation.
    """
    try:
        return _internal_project_to_image_by_rotation(
            mesh._internal_mesh,
            rotation_steps,
            angle_delta_x,
            angle_delta_y,
            angle_delta_z,
            filtering,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")


def project_to_image_by_rotation_to_file(
    mesh: Mesh,
    rotation_steps: int,
    angle_delta_x: float,
    angle_delta_y: float,
    angle_delta_z: float,
    filtering: int = 0,
    verbose: bool = False,
) -> None:
    """
    Generate and save a batch of orthographic projections of the mesh by rotating it in 3D space, writing the resulting images to disk.

    Args:
        mesh (Mesh): The mesh object to project (as returned by `read_mesh`).
        rotation_steps (int): Number of rotation steps to perform. The mesh will be rotated this many times, generating a projection at each step.
        angle_delta_x (float): Rotation angle increment (in radians) around the X axis for each step.
        angle_delta_y (float): Rotation angle increment (in radians) around the Y axis for each step.
        angle_delta_z (float): Rotation angle increment (in radians) around the Z axis for each step.
        filtering (int, optional): Radius for outlier filtering in the depth map (default: 0, disables filtering).
        verbose (bool, optional): If True, enables verbose logging from the Rust backend (default: False).

    Returns:
        None

    Side Effects:
        Saves the generated projection images to disk.

    Raises:
        RuntimeError: If the Rust backend fails during projection generation or file writing.
    """
    try:
        return _internal_project_to_image_by_rotation_to_file(
            mesh._internal_mesh,
            rotation_steps,
            angle_delta_x,
            angle_delta_y,
            angle_delta_z,
            filtering,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during projection generation: {e}")


def project_to_point_cloud_by_sampling(
    mesh: Mesh,
    point_cloud_density_factor: float = 1.0,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a point cloud from the mesh by sampling its surface, returning the 3D points and their corresponding colors.

    Args:
        mesh (Mesh): The mesh object to sample from (as returned by `read_mesh`).
        point_cloud_density_factor (float, optional): Controls the density of the sampled point cloud. Higher values produce more points (default: 1.0).
        verbose (bool, optional): If True, enables verbose logging from the Rust backend (default: False).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - points (np.ndarray): Array of shape (N, 3) containing the 3D coordinates (f32) of the sampled points.
            - colors (np.ndarray): Array of shape (N, 3) containing the RGB color values (uint8) for each point.

    Raises:
        RuntimeError: If the Rust backend fails during point cloud generation.
    """
    try:
        return _internal_project_to_point_cloud_by_sampling(
            mesh._internal_mesh,
            point_cloud_density_factor,
            verbose,
        )
    except Exception as e:
        raise RuntimeError(f"Rust backend failed during point cloud generation: {e}")
