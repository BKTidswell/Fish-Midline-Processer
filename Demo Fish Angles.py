import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use('tkagg')

finding_vals = True

n_fish = 8

# while finding_vals:
#     rand_vals = (np.random.rand(8)-0.5) * 360

#     sin_headings = np.sin(np.deg2rad(rand_vals))
#     cos_headings = np.cos(np.deg2rad(rand_vals))

#     polarization = (1/n_fish)*np.sqrt(np.nansum(sin_headings, axis=0)**2 + np.nansum(cos_headings, axis=0)**2)

#     if round(polarization,2) == 0.75:
#         finding_vals = False

# print(polarization)
# print(rand_vals)

# print("")

# finding_vals = True

# while finding_vals:
#     rand_vals = (np.random.rand(8)-0.5) * 360

#     sin_headings = np.sin(np.deg2rad(rand_vals))
#     cos_headings = np.cos(np.deg2rad(rand_vals))

#     polarization = (1/n_fish)*np.sqrt(np.nansum(sin_headings, axis=0)**2 + np.nansum(cos_headings, axis=0)**2)

#     if round(polarization,2) == 0:
#         finding_vals = False

# print(polarization)
# print(rand_vals)

graph_polar = []

for i in range(100000):
    rand_vals = (np.random.rand(8)-0.5) * 360

    sin_headings = np.sin(np.deg2rad(rand_vals))
    cos_headings = np.cos(np.deg2rad(rand_vals))

    polarization = (1/n_fish)*np.sqrt(np.nansum(sin_headings, axis=0)**2 + np.nansum(cos_headings, axis=0)**2)


    graph_polar.append(polarization)


plt.hist(graph_polar)
plt.show() 

