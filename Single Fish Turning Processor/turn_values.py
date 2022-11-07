import numpy as np
import math

main_vec = [1,0] #From origin to 1,0

#Other points move in a unit circle off of main_vec

deg_0_vec = [1,0]
deg_30_vec = [0.866,0.5]
deg_45_vec = [0.707,0.707]
deg_60_vec = [0.5,0.866]
deg_90_vec = [0,1]
deg_120_vec = [-0.5,0.866]
deg_135_vec = [-0.707,0.707]
deg_150_vec = [-0.866,0.5]
deg_180_vec = [0,0]

other_vecs = [deg_0_vec,deg_30_vec,deg_60_vec,
              deg_45_vec,deg_90_vec,deg_120_vec,
              deg_135_vec,deg_150_vec,deg_180_vec]

for v in other_vecs:

    dot_prod = np.dot(main_vec,v)

    print(dot_prod)