import os, sys
import math
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Ellipse
from fish_core import *

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

# sns.displot(all_cs)
# plt.show()

if flow == "F0":
	graph_color = "Blues"
	detail_color = "red"
else:
	graph_color = "Blues"
	detail_color = "red"

angle_bins = [30]
dist_bins = [1]
colors = ["Blues","winter_r","cool","hot_r","bone_r","plasma_r","GnBu","PuBuGn"]

##HERE
print("Got the data, let's start graphing!")

for angle_bin_size in angle_bins:
	for dist_bin_size in dist_bins:
		for graph_color in colors:

			plt_save_name = "Heatmaps/polar_hmap_{}_{}_{}_{}_{}_{}.png".format(flow,dark,turb,angle_bin_size,dist_bin_size,graph_color)

			print(plt_save_name)

			angles = (np.arctan2(all_ys,all_xs) * 180 / np.pi)
			#See notes, this makes it from 0 to 360
			angles = np.mod(abs(angles-360),360)
			#This rotates it so that 0 is at the top and 180 is below the fish
			angles = np.mod(angles+90,360)

			all_dists = get_dist_np(0,0,all_xs,all_ys)

			#angle_bin_size = 30
			polar_axis = np.linspace(0,360,int(360/angle_bin_size)+1) - angle_bin_size/2
			polar_axis = (polar_axis+angle_bin_size/2) * np.pi /180

			#dist_bin_size = 0.25
			d_range = round_down(np.max(all_dists),base=dist_bin_size)
			d_axis = np.linspace(0,d_range,int(d_range/dist_bin_size)+1)

			polar_array = np.zeros((int(360/angle_bin_size), len(d_axis), len(angles)))

			for i in range(len(angles)):
				a = int(angles[i]/angle_bin_size)
				r = int(round_down(all_dists[i],base=dist_bin_size)/dist_bin_size)
				polar_array[a][r][i] = all_cs[i]


			polar_array[polar_array == 0] = 'nan'
			polar_vals = np.nanmean(polar_array, axis=2)
			#print(polar_vals)
			#polar_vals = np.nan_to_num(polar_vals)
			polar_vals = np.append(polar_vals,polar_vals[0].reshape(1, (len(d_axis))),axis=0)
			
			r, th = np.meshgrid(d_axis, polar_axis)


			fig = plt.figure()
			ax = fig.add_subplot(111, projection='polar')
			plt.pcolormesh(th, r, polar_vals, cmap = graph_color, vmin = -0.75, vmax = 0.75)
			ax.set_xticks(polar_axis)
			ax.set_yticks(d_axis)
			ax.set_theta_zero_location("W")
			ax.set_theta_direction(-1)
			ax.set_thetamin(0)
			ax.set_thetamax(180)
			plt.plot(polar_axis, r, ls='none', color = 'k') 
			plt.grid()
			plt.colorbar()
			#plt.show()

			plt.savefig(plt_save_name)
			plt.close()


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