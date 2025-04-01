import os
import time

import cv2
import numpy as np

import mesh_projector


def save_projections(projections):
    os.makedirs("./projections", exist_ok=True)
    for i in range(len(projections)):
        image = projections[i].astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"projections/projection_{i}.png", image_bgr)


if __name__ == "__main__":
    # mesh_projector parameters
    precision = 12
    filtering = 0
    verbose = False

    # Load mesh through helper function
    obj_path = "assets/just_a_girl_C0-L5_deq_tri.obj"
    tex_path = "assets/just_a_girl_C0-L5_deq_tri_0.png"
    mesh = mesh_projector.read_mesh(obj_path, tex_path, precision, verbose=verbose)

    # Generate projections
    t0 = time.time()
    images, ocp_maps = mesh_projector.generate_projections(
        *mesh, precision, filtering, verbose=verbose
    )
    t1 = time.time()
    print(f"Done. Time taken: {(t1 - t0):.2f} s")

    # Save projections as PNG
    save_projections(images)
