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
    # Project the mesh using rotation and save the images to disk
    mp.project_to_image_by_rotation_to_file(
        mesh,
        rotation_steps=ROTATION_STEPS,
        angle_delta_x=ANGLE_DELTA_X,
        angle_delta_y=ANGLE_DELTA_Y,
        angle_delta_z=ANGLE_DELTA_Z,
        filtering=FILTERING,
        verbose=VERBOSE,
    )
