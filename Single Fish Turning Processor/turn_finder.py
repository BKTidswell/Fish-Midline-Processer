import matplotlib.pyplot as plt
from matplotlib import gridspec 
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pandas
import os
import numpy as np

header = list(range(4))
diff_needed = 15

smooth_window_size = 41

fish_data = pandas.read_csv("Data/2022_06_10_01_LN_DN_F0_V1_02_03_08_17_33_38_44_45.csv",index_col=0, header=header)

scorerer = fish_data.keys()[0][0]

#print(fish_data[scorerer]["individual1"])

x_data = fish_data[scorerer]["individual2"]["head"]["x"]
y_data = fish_data[scorerer]["individual2"]["head"]["y"]

x_data_smooth = savgol_filter(x_data, smooth_window_size, 3)
y_data_smooth = savgol_filter(y_data, smooth_window_size, 3)

x_data_rolled = np.roll(x_data_smooth, 5)
y_data_rolled = np.roll(y_data_smooth, 5)

x_diff = x_data_smooth - x_data_rolled
y_diff = y_data_smooth - y_data_rolled

angle = np.rad2deg(np.arctan2(y_diff,x_diff))
angle_diff = angle - np.roll(angle, 1)
angle_diff = (angle_diff + 180) % 360 - 180

angle_diff = savgol_filter(angle_diff, smooth_window_size, 3)

angle_diff_windowed = abs(np.convolve(angle_diff,np.ones(10,dtype=int),'valid'))

fig = plt.figure(figsize=(8, 6))

gs = gridspec.GridSpec(ncols = 3, nrows = 3) 

peaks, _  = find_peaks(angle_diff_windowed, prominence = diff_needed)

x_data = x_data[:len(angle_diff_windowed)]
y_data = y_data[:len(angle_diff_windowed)]

ax0 = plt.subplot(gs[0,0])
ax0.plot(x_data, y_data)
ax0.plot(x_data_smooth, y_data_smooth)
ax0.scatter(x_data[0], y_data[0])
ax0.plot(x_data[peaks], y_data[peaks], "x")
#ax0.scatter(x_data[abs(angle_diff_windowed) > diff_needed], y_data[abs(angle_diff_windowed) > diff_needed])

axX = plt.subplot(gs[1,0])
axX.plot(np.arange(len(x_data)), x_data)
axX.plot(peaks, x_data[peaks], "x")

axY = plt.subplot(gs[2,0])
axY.plot(np.arange(len(y_data)), y_data)
axY.plot(peaks, y_data[peaks], "x")

ax1 = plt.subplot(gs[:,1])
ax1.scatter(np.arange(len(angle)), angle, s = 2)
ax1.plot(peaks, angle[peaks], "x")

ax2 = plt.subplot(gs[:,2])
ax2.scatter(np.arange(len(angle_diff)), angle_diff, s = 2)
ax2.scatter(np.arange(len(angle_diff_windowed)), angle_diff_windowed, s = 2)
ax2.plot(peaks, angle_diff_windowed[peaks], "x")

plt.show()
