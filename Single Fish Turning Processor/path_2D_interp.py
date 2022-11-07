import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas

##DOES NOT WORK PLS FIX

header = list(range(4))
diff_needed = 15

smooth_window_size = 41

fish_data = pandas.read_csv("Annushka_Data/2021_07_07_16_TN_DY_F2_V1DLC_dlcrnetms5_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv",index_col=0, header=header)

scorerer = fish_data.keys()[0][0]

x_data = fish_data[scorerer]["individual3"]["head"]["x"]
y_data = fish_data[scorerer]["individual3"]["head"]["y"]

points = np.array([x_data.dropna(),
                   y_data.dropna()]).T  # a (nbre_points x nbre_dim) array

# Linear length along the line:
distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]

distance = distance + np.random.rand(len(distance))/1000

# Interpolation for different methods:
interpolations_methods = ['slinear', 'quadratic', 'cubic']
alpha = np.linspace(0, 1, 75)

interpolated_points = {}
for method in interpolations_methods:
    interpolator =  interp1d(distance, points, kind=method, axis=0)
    interpolated_points[method] = interpolator(alpha)

# Graph:
plt.figure(figsize=(7,7))
for method_name, curve in interpolated_points.items():
    plt.plot(*curve.T, '-', label=method_name)

plt.plot(*points.T, 'ok', label='original points')
plt.axis('equal'); plt.legend(); plt.xlabel('x'); plt.ylabel('y')