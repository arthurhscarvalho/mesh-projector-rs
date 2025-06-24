import os

import cv2
import numpy as np

import mesh_projector as mp

if __name__ == "__main__":
    # Configuration
    OBJ_PATH = "assets/just_a_girl_C0-L5_deq_tri.obj"
    TEXTURE_PATH = "assets/just_a_girl_C0-L5_deq_tri_0.png"
    MTL_PATH = "assets/just_a_girl_C0-L5_deq_tri.mtl"
    PRECISION = 12  # This results in 4096x4096 images
    FLIP_UV = True
    IGNORE_OPACITY = True
    VERBOSE = True
    FILTERING = 0
    # Rotation parameters
    ROTATION_STEPS = 16  # Number of rotation steps
    ANGLE_DELTA_X = np.pi / 16  # Small rotation around X axis
    ANGLE_DELTA_Y = np.pi / 8  # Rotation around Y axis for different views
    ANGLE_DELTA_Z = 0.0  # No rotation around Z axis
    # Read the mesh
    mesh = mp.read_mesh(
        obj_path=OBJ_PATH,
        texture_path=TEXTURE_PATH,
        mtl_path=MTL_PATH,
        precision=PRECISION,
        flip_uv=FLIP_UV,
        ignore_opacity=IGNORE_OPACITY,
        verbose=VERBOSE,
    )
    # Project the mesh using rotation
    projections, depth_maps, normal_maps, occupancy_maps = (
        mp.project_to_image_by_rotation(
            mesh,
            rotation_steps=ROTATION_STEPS,
            angle_delta_x=ANGLE_DELTA_X,
            angle_delta_y=ANGLE_DELTA_Y,
            angle_delta_z=ANGLE_DELTA_Z,
            filtering=FILTERING,
            verbose=VERBOSE,
        )
    )
    # Create a directory to save the images
    os.makedirs("rotation_projections_example", exist_ok=True)
    # Save the projections to png files
    for i, projection in enumerate(projections):
        cv2.imwrite(
            f"rotation_projections_example/proj_{i:02d}.png",
            cv2.cvtColor(projection, cv2.COLOR_RGB2BGR),
        )
    # Save the depth maps to png files
    for i, depth_map in enumerate(depth_maps):
        cv2.imwrite(
            f"rotation_projections_example/depth_map_{i:02d}.png",
            cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR),
        )
    # Save the normal maps to png files
    for i, normal_map in enumerate(normal_maps):
        cv2.imwrite(
            f"rotation_projections_example/normal_map_{i:02d}.png",
            cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR),
        )
    # Save the occupancy maps to png files
    for i, occupancy_map in enumerate(occupancy_maps):
        cv2.imwrite(
            f"rotation_projections_example/ocp_map_{i:02d}.png",
            cv2.cvtColor(occupancy_map, cv2.COLOR_RGB2BGR),
        )
    print("Generated projections saved to 'rotation_projections_example/' directory")
