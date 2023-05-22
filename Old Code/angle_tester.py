import numpy as np


def calc_heading_diff(f1_vec,f2_vec):

  f1_heading = np.rad2deg(np.arctan2(f1_vec[1],f1_vec[0]))
  f2_heading = np.rad2deg(np.arctan2(f2_vec[1],f2_vec[0]))

  heading_diff = np.rad2deg(np.arctan2(np.sin(np.deg2rad(f1_heading-f2_heading)),
                                       np.cos(np.deg2rad(f1_heading-f2_heading))))

  return heading_diff


def angle_dot(f1_vec,f2_vec):
    dot_product = np.dot(f1_vec, f2_vec)
    prod_of_norms = np.linalg.norm(f1_vec) * np.linalg.norm(f2_vec)
    angle = (np.degrees(np.arccos(dot_product / prod_of_norms)))

    print(dot_product,prod_of_norms)
    return angle

f1_vec = np.asarray([-1,1])
f2_vec = np.asarray([0,0])

print(calc_heading_diff(f1_vec,f2_vec))
print(angle_dot(f1_vec,f2_vec))
