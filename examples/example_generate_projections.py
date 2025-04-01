import cv2
import numpy as np
import mesh_projector
import time


def save_projections(projections):
    for i in range(len(projections)):
        image = projections[i].astype(np.uint8)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"projection_{i}.png", image_bgr)


# mesh_projector parameters
precision = 12
filtering = 0
save = True

# Load mesh through helper function
obj_path = "assets/just_a_girl_C0-L5_deq_tri.obj"
tex_path = "assets/just_a_girl_C0-L5_deq_tri_0.png"
# obj_path = "/home/arthurc/meshes/just_a_girl-distorted/just_a_girl/just_a_girl_dec0.40_qp10_qt9_cqlevel_63.obj"
# tex_path = "/home/arthurc/meshes/just_a_girl-distorted/just_a_girl/just_a_girl_dec0.40_qp10_qt9_cqlevel_63_0.png"
# obj_path = "/home/arthurc/meshes/SJTU-TMQA/reference_dataset/airplane/airplane.obj"
# tex_path = "/home/arthurc/meshes/SJTU-TMQA/reference_dataset/airplane/airplane.jpg"
# obj_path = "/home/arthurc/meshes/SJTU-TMQA/reference_dataset/deadRose/deadRose.obj"
# tex_path = "/home/arthurc/meshes/SJTU-TMQA/reference_dataset/deadRose/deadRose.jpg"
mesh = mesh_projector.read_mesh(obj_path, tex_path, precision)


# Generate projections
t0 = time.time()
images, ocp_maps = mesh_projector.generate_projections(*mesh, precision, filtering)
t1 = time.time()
print(f"Done. Time taken: {(t1 - t0):.2f} s")

# Save projections as PNG
if save:
    save_projections(images)
