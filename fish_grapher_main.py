import os
import math
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from fish_core import *
import pandas as pd
import random

data_folder = os.getcwd()+"/Finished_Fish_Data/"
flow = "F2"
dark = "DN"
turb = "TN"

save_file = "data_{}_{}_{}.npy".format(flow,dark,turb)

new = False

#6/29/20 ~700 pixels = 24.5 cm bc of curvature: 700/24.5 pixels per cm
cnvrt_dict = {"2020":{"06": {"29":700/24.5},
					  "07": {"28":800/24.5}}}

def cnvrt_pixels(year,month,day,distance):
	return distance/cnvrt_dict[year][month][day]

num_data = 0
data_files = []

for file_name in os.listdir(data_folder):
	if file_name.endswith(".csv") and flow in file_name and dark in file_name and turb in file_name:
		num_data += 1
		data_files.append(file_name)

all_xs = []
all_ys = []
all_cs = []

if new:
	for file_name in data_files:
		year = file_name[0:4]
		month = file_name[5:7]
		day = file_name[8:10]

		print(year,month,day,flow,dark,turb)

		#Create the fish dict and get the time points
		fish_dict,time_points = DLC_CSV_to_dict(num_fish = n_fish, fish_parts = b_parts_csv, file = data_folder+file_name)

		fish_para = []
		fish_perp = []
		fish_paths = []

		#For each fish get the para and perp distances and append to the array
		for i in range(n_fish):
			f_para_temp,f_perp_temp,body_line = generate_midline(fish_dict[i],time_points)

			fish_para.append(f_para_temp)
			fish_perp.append(f_perp_temp)
			fish_paths.append(body_line)

		fish_para = np.asarray(fish_para)
		fish_perp = np.asarray(fish_perp)

		#Ok So now I want to create a heatmaps of those slopes of the Hilbert Phases over time
		#Let's just start with a heatmap of fish position over time, centered around individual 1

		#This is the big array that will be turned into a heat map


		#First create an n_fish x n_fish x timepoints array to store the slopes in

		slope_array = np.zeros((n_fish,n_fish,time_points))

		for i in range(n_fish):
			for j in range(n_fish):

				#Get the signal for each with Hilbert phase
				analytic_signal_main = hilbert(normalize_signal(fish_perp[i][:,5]))
				instantaneous_phase_main = np.unwrap(np.angle(analytic_signal_main))

				analytic_signal = hilbert(normalize_signal(fish_perp[j][:,5]))
				instantaneous_phase = np.unwrap(np.angle(analytic_signal))

				# #Now get the slope
				# dx = np.diff(instantaneous_phase_main)
				# dy = np.diff(instantaneous_phase)

				#This normalizes from 0 to 1. Not sure I should do this, but here we are
				#If I don't it really throws off the scale.

				#10/13 slope is now 0 when they are aligned and higher when worse. 

				#10/16 uses the get slope function for smoother slope
				slope = get_slope(instantaneous_phase_main,instantaneous_phase)
				norm_slope = abs(slope-1)

				#Now copy it all over. Time is reduced becuase diff makes it shorter
				for t in range(time_points-5):
					slope_array[i][j][t] = norm_slope[t]

		dim = 3000
		offset = dim/2
		mean_hmap = True

		time_pos_array = np.zeros((dim,dim,time_points))

		if mean_hmap:
			time_pos_array[time_pos_array == 0] = np.NaN


		fish_head_xs = []
		fish_head_ys = []

		for i in range(n_fish):
			fish_head_xs.append(fish_dict[i]["head"]["x"])
			fish_head_ys.append(fish_dict[i]["head"]["y"])

		fish_head_xs = np.asarray(fish_head_xs)
		fish_head_ys = np.asarray(fish_head_ys)

		#Go through all timepoints with each fish as the center one
		for f in range(n_fish):

			for i in range(time_points):
				main_fish_x = fish_head_xs[f][i]
				main_fish_y = fish_head_ys[f][i]

				#This prevents perfect symetry and doubling up on fish
				for j in range(f+1,n_fish):
					other_fish_x = fish_head_xs[j][i]
					other_fish_y = fish_head_ys[j][i]

					#This order is so that the heatmap faces correctly upstream
					x_diff = cnvrt_pixels(year,month,day,main_fish_x - other_fish_x)
					y_diff = cnvrt_pixels(year,month,day,other_fish_y - main_fish_y)

					x_pos = int(x_diff+offset)
					y_pos = int(y_diff+offset)

					all_xs.append(x_diff)
					all_ys.append(y_diff)

					all_xs.append(-1*x_diff)
					all_ys.append(y_diff)

					all_cs.append(slope_array[f][j][i])
					all_cs.append(slope_array[f][j][i])

					if mean_hmap:
						time_pos_array[y_pos][x_pos][i] = slope_array[f][j][i]
					else:
						time_pos_array[y_pos][x_pos][i] = 1

		# if mean_hmap:
		# 	heatmap_array = np.nanmean(time_pos_array, axis=2)
		# else:
		# 	heatmap_array = np.sum(time_pos_array, axis=2)

		# #remove the center point for scaling:
		# heatmap_array[int(dim/2)][int(dim/2)] = 0

		# new_dim = 100

		# if mean_hmap:
		# 	shrunk_map = shrink_nanmean(heatmap_array,new_dim,new_dim)
		# else:
		# 	shrunk_map = shrink_sum(heatmap_array,new_dim,new_dim)

		# shrunk_map[shrunk_map == 0] = np.NaN

		# fig, ax = plt.subplots()
		# ax.set_ylim(0,new_dim-1)
		# im = ax.imshow(shrunk_map,cmap='jet')
		# im.set_clim(0,75)
		# fig.colorbar(im)
		# plt.show()

	all_xs = np.asarray(all_xs)
	all_ys = np.asarray(all_ys)
	all_cs = np.asarray(all_cs)

	with open(save_file, 'wb') as f:
		np.save(f, all_xs)
		np.save(f, all_ys)
		np.save(f, all_cs)

def round_down(x, base=5):
	if x < 0:
		return base * math.ceil(x/base)
	else:
		return base * math.floor(x/base)
	#return base * math.floor(x/base)

if not new:
	with open(save_file, 'rb') as f:
		all_xs = np.load(f)
		all_ys = np.load(f)
		all_cs = np.load(f)

# sns.set_style("white")
# sns.kdeplot(x=all_xs, y=all_ys, cmap="Blues", shade=True, bw_method=.15)
# plt.scatter(x=0, y=0, color='r')
# plt.show()

# sns.displot(all_cs)
# plt.show()

bin_size = 3

x_range = round_down(np.max(np.absolute(all_xs)),base=bin_size)
y_range = round_down(np.max(np.absolute(all_ys)),base=bin_size)

#x and y are swapped to make it graph right
heatmap_array = np.zeros((int(y_range*2/bin_size)+1,int(x_range*2/bin_size)+1,len(all_xs)))

x_axis = np.asarray(range(-1*x_range,x_range+bin_size,bin_size))
y_axis = np.asarray(range(-1*y_range,y_range+bin_size,bin_size))

x_offset = int(x_range/bin_size)
y_offset = int(y_range/bin_size)

rounded_xs = []

for i in range(len(all_xs)):
	x = int(round_down(all_xs[i],base=bin_size)/bin_size + x_offset)
	y = int(round_down(all_ys[i],base=bin_size)/bin_size + y_offset)

	heatmap_array[y][x][i] = all_cs[i]
	rounded_xs.append(x)

heatmap_array[heatmap_array == 0] = 'nan'
mean_map = np.nanmean(heatmap_array, axis=2)
mean_map = np.nan_to_num(mean_map)

#Makes maps that work with ax.contour so that x and y axis are repeated over the Z array
#https://alex.miller.im/posts/contour-plots-in-python-matplotlib-x-y-z/
x_map = np.repeat(x_axis.reshape(1,len(x_axis)),len(y_axis),axis=0)
y_map = np.repeat(y_axis.reshape(len(y_axis),1),len(x_axis),axis=1)


fig = plt.figure()
ax = fig.add_subplot(111)
# Generate a contour plot
cp = ax.contourf(x_map, y_map, mean_map, cmap = "Blues_r", cmin=0,cmax=2)
cbar = fig.colorbar(cp)
cp.set_clim(0, 2)
ax.plot(0, 0, 'ro')
plt.show()

# fig, ax = plt.subplots()
# im = ax.imshow(mean_map, cmap = "Blues_r")
# plt.show()

# print(sum(all_cs)/len(all_cs))
# n = round(len(all_cs)/2)
# print(sorted(all_cs)[n])

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_trisurf(all_xs, all_ys, all_cs, cmap=plt.cm.viridis, linewidth=0.2)
# plt.show()

# sns.set(style="white", color_codes=True)
# sns.jointplot(x=all_xs, y=all_ys, kind='kde', color="skyblue")
# plt.show()


#plot_fish_vid(fish_dict,fish_para,fish_perp,fish_paths,time_points)	