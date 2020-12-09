import os, sys
import math
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Ellipse
from fish_core import *
from PIL import Image
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

data_folder = os.getcwd()+"/Finished_Fish_Data/"
flow = "F2"
dark = "DN"
turb = "TN"

save_file = "data_{}_{}_{}.npy".format(flow,dark,turb)

new = False

num_data = 0
data_files = []

for file_name in os.listdir(data_folder):
	if file_name.endswith(".csv") and flow in file_name and dark in file_name and turb in file_name:
		num_data += 1
		data_files.append(file_name)

all_xs = []
all_ys = []
all_cs = []

for file_name in data_files:
	year = file_name[0:4]
	month = file_name[5:7]
	day = file_name[8:10]
	trial = file_name[11:13]

	print(year,month,day,trial,flow,dark,turb)

	#Create the fish dict and get the time points
	fish_dict,time_points = DLC_CSV_to_dict(num_fish = n_fish, fish_parts = b_parts_csv, file = data_folder+file_name)

	fish_para = []
	fish_perp = []
	fish_paths = []

	#Find the pixel to bodylength conversion to normalize all distances by body length
	cnvrt_pix_bl = []

	for i in range(n_fish):
		cnvrt_pix_bl.append(median_fish_len(fish_dict,i))

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
				x_diff = (main_fish_x - other_fish_x)/cnvrt_pix_bl[f]
				y_diff = (other_fish_y - main_fish_y)/cnvrt_pix_bl[f]

				# x_pos = int(x_diff+offset)
				# y_pos = int(y_diff+offset)

				if abs(x_diff) > 7 or abs(y_diff) > 7:
					pass

				else:
					all_xs.append(x_diff)
					all_ys.append(y_diff)

					all_xs.append(-1*x_diff)
					all_ys.append(y_diff)

					# -1 * log(x+1)+1
					all_cs.append(-1*math.log(slope_array[f][j][i]+1)+1)
					all_cs.append(-1*math.log(slope_array[f][j][i]+1)+1)

					#time_pos_array[y_pos][x_pos][i] = 1

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

# def round_down(x, base=5):
# 	if x < 0:
# 		return base * math.ceil(x/base)
# 	else:
# 		return base * math.floor(x/base)
# 	#return base * math.floor(x/base)

# if not new:
# 	with open(save_file, 'rb') as f:
# 		all_xs = np.load(f)
# 		all_ys = np.load(f)
# 		all_cs = np.load(f)

# sns.displot(all_cs)
# plt.show()

##HERE
#print("Got the data, let's start graphing!")

#Replaced with polar plots

# mean_xs = np.mean(all_xs[::2])
# mean_ys = np.mean(all_ys)

# sd_xs = np.std(all_xs)
# sd_ys = np.std(all_ys)

# sns.set_style("white")
# ax = sns.kdeplot(x=all_xs, y=all_ys, cmap="GnBu", shade=True, bw_method=.15)
# plt.scatter(x=0, y=0, color='r')
# plt.gcf().gca().add_artist(Ellipse((0, 0),mean_xs,mean_ys,facecolor="none",edgecolor='white'))
# plt.gcf().gca().add_artist(Ellipse((0, 0),sd_xs,sd_ys,facecolor="none",edgecolor="red"))
# plt.show()

# bin_size = 0.25

# x_range = round_down(np.max(np.absolute(all_xs)),base=bin_size)
# y_range = round_down(np.max(np.absolute(all_ys)),base=bin_size)

# #x and y are swapped to make it graph right
# heatmap_array = np.zeros((int(y_range*2/bin_size)+1,int(x_range*2/bin_size)+1,len(all_xs)))

# x_axis = np.linspace(-1*x_range,x_range,int(x_range*2/bin_size)+1)
# y_axis = np.linspace(-1*y_range,y_range,int(y_range*2/bin_size)+1)

# x_offset = int(x_range/bin_size)
# y_offset = int(y_range/bin_size)

# for i in range(len(all_xs)):
# 	x = int(round_down(all_xs[i],base=bin_size)/bin_size + x_offset)
# 	y = int(round_down(all_ys[i],base=bin_size)/bin_size + y_offset)

# 	heatmap_array[y][x][i] = all_cs[i]

# heatmap_array[heatmap_array == 0] = 'nan'
# mean_map = np.nanmean(heatmap_array, axis=2)
# mean_map = np.nan_to_num(mean_map)

# #Makes maps that work with ax.contour so that x and y axis are repeated over the Z array
# #https://alex.miller.im/posts/contour-plots-in-python-matplotlib-x-y-z/
# x_map = np.repeat(x_axis.reshape(1,len(x_axis)),len(y_axis),axis=0)
# y_map = np.repeat(y_axis.reshape(len(y_axis),1),len(x_axis),axis=1)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # Generate a contour plot
# cp = ax.contourf(x_map, y_map, mean_map, cmap = "GnBu")
# cbar = fig.colorbar(cp)
# ax.plot(0, 0, 'ro')
# plt.show()

# fig, ax = plt.subplots()
# im = ax.imshow(mean_map, cmap = "Blues_r")
# plt.show()

# angles = (np.arctan2(all_ys,all_xs) * 180 / np.pi)
# #See notes, this makes it from 0 to 360
# angles = np.mod(abs(angles-360),360)
# #This rotates it so that 0 is at the top and 180 is below the fish
# angles = np.mod(angles+90,360)

# all_dists = get_dist_np(0,0,all_xs,all_ys)

# # print(all_dists.shape)
# # print(angles.shape)

# angle_bin_size = 30
# polar_axis = np.linspace(0,360,int(360/angle_bin_size)+1) - angle_bin_size/2
# polar_axis = (polar_axis+angle_bin_size/2) * np.pi /180

# dist_bin_size = 1
# d_range = round_down(np.max(all_dists),base=dist_bin_size)
# #print(polar_axis)
# d_axis = np.linspace(0,d_range,int(d_range/dist_bin_size)+1)

# # print(d_axis)

# polar_array = np.zeros((int(360/angle_bin_size), len(d_axis), len(angles)))
# polar_density_array = np.zeros((int(360/angle_bin_size), len(d_axis), len(angles)))

# for i in range(len(angles)):
# 	a = int(angles[i]/angle_bin_size)
# 	r = int(round_down(all_dists[i],base=dist_bin_size)/dist_bin_size)
# 	polar_array[a][r][i] = all_cs[i]
# 	polar_density_array[a][r][i] = 1


# polar_array[polar_array == 0] = 'nan'
# polar_vals = np.nanmean(polar_array, axis=2)
# #print(polar_vals)
# #polar_vals = np.nan_to_num(polar_vals)
# polar_vals = np.append(polar_vals,polar_vals[0].reshape(1, (len(d_axis))),axis=0)

# #print(polar_vals.shape)

# r, th = np.meshgrid(d_axis, polar_axis)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='polar')
# plt.pcolormesh(th, r, polar_vals, cmap = "GnBu", vmin = -0.75, vmax = 0.75)

# arr_png = mpimg.imread('fish.png')
# imagebox = OffsetImage(arr_png, zoom = 0.65)
# ab = AnnotationBbox(imagebox, (0, 0), frameon = False)
# ax.add_artist(ab)

# ax.set_xticks(polar_axis)
# ax.set_yticks(d_axis)
# ax.set_theta_zero_location("W")
# ax.set_theta_direction(-1)
# ax.set_thetamin(0)
# ax.set_thetamax(180)

# plt.plot(polar_axis, r, ls='none', color = 'k') 
# plt.grid()
# plt.colorbar()
# plt.show()


# polar_density = np.sum(polar_density_array, axis=2)
# polar_density = polar_density/np.amax(polar_density)
# polar_density = np.append(polar_density,polar_density[0].reshape(1, (len(d_axis))),axis=0)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='polar')
# plt.pcolormesh(th, r, polar_density, cmap = "GnBu", vmin = 0, vmax = 1)

# arr_png = mpimg.imread('fish.png')
# imagebox = OffsetImage(arr_png, zoom = 0.65)
# ab = AnnotationBbox(imagebox, (0, 0), frameon = False)
# ax.add_artist(ab)

# ax.set_xticks(polar_axis)
# ax.set_yticks(d_axis)
# ax.set_theta_zero_location("W")
# ax.set_theta_direction(-1)
# ax.set_thetamin(0)
# ax.set_thetamax(180)

# plt.plot(polar_axis, r, ls='none', color = 'k') 
# plt.grid()
# plt.colorbar()
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='polar')
# plt.polar(polar_axis,polar_vals,'b-',label='No Flow')
# ax.set_theta_zero_location("W")
# ax.set_theta_direction(-1)
# ax.set_thetamin(0)
# ax.set_thetamax(180)
# plt.legend(loc=(1,1))
# #plt.legend([no_flow, flow],["No Flow","Flow"])
# plt.show()

##TO HERE

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