import matplotlib.pyplot as plt
from matplotlib import gridspec 
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pandas
import os
import numpy as np
import math

def roundnearest(x, base=25):
    return base * round(x/base)

def calc_mag(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def moving_sum(x, w):
    return np.convolve(x, np.ones(w), 'valid')


header = list(range(4))
diff_needed = 15

smooth_window_size = 13

fish_data = pandas.read_csv("Single_Fish_Data/2022_06_10_01_LN_DN_F2_V1_05_06_13_14_30_47_48_50.csv",index_col=0, header=header)

#fish_data = pandas.read_csv("Annushka_Data/2021_07_07_16_TN_DY_F2_V1DLC_dlcrnetms5_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv",index_col=0, header=header)

scorerer = fish_data.keys()[0][0]

#print(fish_data[scorerer]["individual1"])

x_data = fish_data[scorerer]["individual5"]["head"]["x"].to_numpy()
y_data = fish_data[scorerer]["individual5"]["head"]["y"].to_numpy()

#x_data = savgol_filter(x_data, smooth_window_size, 3)
#y_data = savgol_filter(y_data, smooth_window_size, 3)

# x_data = x_data[::20].copy()
# y_data = y_data[::20].copy()

#print(x_data)
#print(len(x_data))

point_data = np.column_stack((x_data, y_data))

#print(point_data)

#offset = 10

common_peaks = dict()

#Lower offsets find smaller turns
#Higher offsets find bigger turns

#maxoffset = 20 

dot_prods = np.zeros(len(point_data))+1

#for offset in range(2,maxoffset):

offset = 20

for i in range(offset,len(point_data)-offset-1):

    #print(i)

    vec1 = (point_data[i] - point_data[i-offset]) / calc_mag(point_data[i],point_data[i-offset])
    vec2 = (point_data[i+offset] - point_data[i]) / calc_mag(point_data[i+offset],point_data[i])

    dot_prods[i] = np.dot(vec1,vec2)

    if np.isnan(np.dot(vec1,vec2)):
        dot_prods[i] = 1

    #print(dot_prods)


#Trim the edges
#dot_prods = dot_prods[offset:-offset-1]

#print(dot_prods)

dot_prods = abs(dot_prods-1)

#dot_prods_windowed = moving_sum(dot_prods,1)

#print(dot_prods_windowed)

#peak_prom = np.std(dot_prods)*1.5

#0.5 is a 45 degree turn

peak_min = 0.3

dot_prods_over_min = np.where(dot_prods<=peak_min,0,1)

peaks, _  = find_peaks(dot_prods_over_min, prominence = peak_min)

    # for p in peaks:
    #     round_p = roundnearest(p)

    #     if round_p in common_peaks.keys():
    #         common_peaks[round_p] += 1
    #     else:
    #         common_peaks[round_p] = 1

#print(dict(sorted(common_peaks.items())))

#print(dot_prods)

print(peaks)

#Works to put it on the right point on the turn
#peaks = peaks

col = np.where(dot_prods<peak_min,'k','r')

fig = plt.figure(figsize=(8, 6))

gs = gridspec.GridSpec(ncols = 2, nrows = 2) 

ax0 = plt.subplot(gs[:,0])
ax0.plot(x_data, y_data)
# ax0.plot(fish_data[scorerer]["individual2"]["head"]["x"], fish_data[scorerer]["individual2"]["head"]["y"])
# ax0.plot(fish_data[scorerer]["individual3"]["head"]["x"], fish_data[scorerer]["individual3"]["head"]["y"])
# ax0.plot(fish_data[scorerer]["individual4"]["head"]["x"], fish_data[scorerer]["individual4"]["head"]["y"])
# ax0.plot(fish_data[scorerer]["individual5"]["head"]["x"], fish_data[scorerer]["individual5"]["head"]["y"])
# ax0.plot(fish_data[scorerer]["individual6"]["head"]["x"], fish_data[scorerer]["individual6"]["head"]["y"])
# ax0.plot(fish_data[scorerer]["individual7"]["head"]["x"], fish_data[scorerer]["individual7"]["head"]["y"])
# ax0.plot(fish_data[scorerer]["individual8"]["head"]["x"], fish_data[scorerer]["individual8"]["head"]["y"])
ax0.plot(x_data[peaks], y_data[peaks], "x")

# for p in peaks:
#     ax0.plot([x_data[p],x_data[p+offset]],
#              [y_data[p],y_data[p+offset]])
#     ax0.plot([x_data[p+offset],x_data[p+offset*2]],
#              [y_data[p+offset],y_data[p+offset*2]])

# axX = plt.subplot(gs[1,0])
# axX.plot(np.arange(len(x_data)), x_data)
# axX.plot(peaks, x_data[peaks], "x")

# axY = plt.subplot(gs[2,0])
# axY.plot(np.arange(len(y_data)), y_data)
# axY.plot(peaks, y_data[peaks], "x")

ax1 = plt.subplot(gs[:,1])

#ax1.scatter(np.arange(len(dot_prods_windowed)), dot_prods_windowed, c = col)
ax1.plot(np.arange(len(dot_prods)), dot_prods)
ax1.plot(np.arange(len(dot_prods_over_min)), dot_prods_over_min)
#Works best to display
ax1.plot(peaks, dot_prods_over_min[peaks], "x")

plt.show()






