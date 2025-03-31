import cv2
import numpy as np
import mesh_projector
import time


def save_projections(projections):
    for i in range(len(projections)):
        image = projections[i].astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"projection_{i}.png", image_bgr)


# Load the point cloud through open3d
obj_path = "/home/arthurc/meshes/just_a_girl-reference/just_a_girl/just_a_girl_C0-L5_deq_tri.obj"
tex_path = "/home/arthurc/meshes/just_a_girl-reference/just_a_girl/just_a_girl_C0-L5_deq_tri_0.png"
# obj_path = "/home/arthurc/meshes/SJTU-TMQA/reference_dataset/airplane/airplane.obj"
# tex_path = "/home/arthurc/meshes/SJTU-TMQA/reference_dataset/airplane/airplane.jpg"
precision = 12
mesh = mesh_projector.read_mesh(obj_path, tex_path, precision)

# orthographic_projector parameters
filtering = 0
save = True

t0 = time.time()
# The generate_projections function can be used for generating the projections
images, ocp_maps = mesh_projector.generate_projections(*mesh, precision, filtering)
t1 = time.time()
print(f"Done. Time taken: {(t1 - t0):.2f} s")

# The save_projections function is just an example intended for visualization of the results
if save:
    save_projections(images)
