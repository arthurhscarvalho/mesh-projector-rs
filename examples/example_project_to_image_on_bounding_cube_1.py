import os

import cv2

import mesh_projector as mp

if __name__ == "__main__":
    # Configuration
    OBJ_PATH = "assets/Chameleon_Ref_0.obj"
    PRECISION = 12  # This results in 4096x4096 images
    FLIP_UV = True
    IGNORE_OPACITY = True
    VERBOSE = True
    FILTERING = 0
    # Read the mesh (obj-only mode, no texture or mtl file is needed)
    # The renderer will not use any textures, only the geometry and triangle colors,
    # and no shading is applied
    mesh = mp.read_mesh(
        obj_path=OBJ_PATH,
        precision=PRECISION,
        flip_uv=FLIP_UV,
        ignore_opacity=IGNORE_OPACITY,
        verbose=VERBOSE,
    )
    # Project the mesh to the bounding cube
    projections, depth_maps, normal_maps, occupancy_maps = (
        mp.project_to_image_on_bounding_cube(
            mesh,
            filtering=FILTERING,
            verbose=VERBOSE,
        )
    )
    # Create a directory to save the images
    os.makedirs("cube_projections", exist_ok=True)
    # Save the projections to a png file
    for i, projection in enumerate(projections):
        cv2.imwrite(
            f"cube_projections/proj_{i}.png",
            cv2.cvtColor(projection, cv2.COLOR_RGB2BGR),
        )
    # Save the depth maps to a png file
    for i, depth_map in enumerate(depth_maps):
        cv2.imwrite(
            f"cube_projections/depth_map_{i}.png",
            cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR),
        )
    # Save the normal maps to a png file
    for i, normal_map in enumerate(normal_maps):
        cv2.imwrite(
            f"cube_projections/normal_map_{i}.png",
            cv2.cvtColor(normal_map, cv2.COLOR_RGB2BGR),
        )
    # Save the occlusion maps to a png file
    for i, occlusion_map in enumerate(occupancy_maps):
        cv2.imwrite(
            f"cube_projections/ocp_map_{i}.png",
            cv2.cvtColor(occlusion_map, cv2.COLOR_RGB2BGR),
        )
