import os
import math
import seaborn as sns
from scipy import stats
from fish_core import *

data_folder = os.getcwd()+"/Finished_Fish_Data/"

def round_down(x, base=5):
	if x < 0:
		return base * math.ceil(x/base)
	else:
		return base * math.floor(x/base)

#Condition 1
flow_1 = "F0"
dark_1 = "DN"
turb_1 = "TN"

#Condition 2
flow_2 = "F2"
dark_2 = "DN"
turb_2 = "TN"

save_file_1 = "data_{}_{}_{}.npy".format(flow_1,dark_1,turb_1)
save_file_2 = "data_{}_{}_{}.npy".format(flow_2,dark_2,turb_2)

with open(save_file_1, 'rb') as f_1:
	all_xs_1 = np.load(f_1)
	all_ys_1 = np.load(f_1)
	all_cs_1 = np.load(f_1)

with open(save_file_2, 'rb') as f_2:
	all_xs_2 = np.load(f_2)
	all_ys_2 = np.load(f_2)
	all_cs_2 = np.load(f_2)

bin_size = 1

x_range = max(round_down(np.max(np.absolute(all_xs_1)),base=bin_size),
			  round_down(np.max(np.absolute(all_xs_2)),base=bin_size))
y_range = max(round_down(np.max(np.absolute(all_ys_1)),base=bin_size),
			  round_down(np.max(np.absolute(all_ys_2)),base=bin_size))

heatmap_array_1 = np.zeros((int(y_range*2/bin_size)+1,int(x_range*2/bin_size)+1,len(all_xs_1)))
heatmap_array_2 = np.zeros((int(y_range*2/bin_size)+1,int(x_range*2/bin_size)+1,len(all_xs_2)))

x_axis = np.linspace(-1*x_range,x_range,int(x_range*2/bin_size)+1)
y_axis = np.linspace(-1*y_range,y_range,int(y_range*2/bin_size)+1)

x_offset = int(x_range/bin_size)
y_offset = int(y_range/bin_size)

for i in range(len(all_xs_1)):
	x_1 = int(round_down(all_xs_1[i],base=bin_size)/bin_size + x_offset)
	y_1 = int(round_down(all_ys_1[i],base=bin_size)/bin_size + y_offset)

	heatmap_array_1[y_1][x_1][i] = all_cs_1[i]

for i in range(len(all_xs_2)):
	x_2 = int(round_down(all_xs_2[i],base=bin_size)/bin_size + x_offset)
	y_2 = int(round_down(all_ys_2[i],base=bin_size)/bin_size + y_offset)

	heatmap_array_2[y_2][x_2][i] = all_cs_2[i]


heatmap_array_1[heatmap_array_1 == 0] = 'nan'
mean_map_1 = np.nanmean(heatmap_array_1, axis=2)
mean_map_1 = np.nan_to_num(mean_map_1)

heatmap_array_2[heatmap_array_2 == 0] = 'nan'
mean_map_2 = np.nanmean(heatmap_array_2, axis=2)
mean_map_2 = np.nan_to_num(mean_map_2)

#Get SD
se_array_1 = stats.sem(heatmap_array_1, axis=2, nan_policy = "omit")
se_array_1 = np.nan_to_num(np.asarray(se_array_1))

se_array_2 = stats.sem(heatmap_array_2, axis=2, nan_policy = "omit")
se_array_2 = np.nan_to_num(np.asarray(se_array_2))

#See if these are all that different
mean_diff_array = abs(mean_map_1 - mean_map_2)
comp_error_array = se_array_1 - se_array_2

#See if the total difference is les than combined error
diff_array = mean_diff_array > comp_error_array

pos_neg_diff_array = np.where(mean_map_1 - mean_map_2 < 0, -1, 1)
#pos_neg_diff_array = pos_neg_diff_array[pos_neg_diff_array > 0] = 1

sig_diff_array = pos_neg_diff_array*diff_array
sig_diff_array = sig_diff_array.astype('float')
sig_diff_array[sig_diff_array == 0] = 'nan'

#Makes maps that work with ax.contour so that x and y axis are repeated over the Z array
#https://alex.miller.im/posts/contour-plots-in-python-matplotlib-x-y-z/
x_map = np.repeat(x_axis.reshape(1,len(x_axis)),len(y_axis),axis=0)
y_map = np.repeat(y_axis.reshape(len(y_axis),1),len(x_axis),axis=1)

diff_map = mean_map_1 - mean_map_2

# fig = plt.figure()
# ax = fig.add_subplot(111)
# # Generate a contour plot
# cp = ax.contourf(x_map, y_map, sig_diff_array, cmap = "bwr_r")
# cbar = fig.colorbar(cp)
# ax.plot(0, 0, 'ko')
# plt.show()

fig = plt.imshow(sig_diff_array, cmap='bwr_r')
plt.show()

#Making polar plots

angles_1 = (np.arctan2(all_ys_1,all_xs_1) * 180 / np.pi)
#See notes, this makes it from 0 to 360
angles_1 = np.mod(abs(angles_1-360),360)
#This rotates it so that 0 is at the top and 180 is below the fish
angles_1 = np.mod(angles_1+90,360)

angles_2 = (np.arctan2(all_ys_2,all_xs_2) * 180 / np.pi)
#See notes, this makes it from 0 to 360
angles_2 = np.mod(abs(angles_2-360),360)
#This rotates it so that 0 is at the top and 180 is below the fish
angles_2 = np.mod(angles_2+90,360)

angle_bin_size = 30
polar_axis = np.linspace(0,360,int(360/angle_bin_size)+1)
polar_axis = (polar_axis+angle_bin_size/2) * np.pi /180

polar_array_1 = np.zeros((int(360/angle_bin_size), len(angles_1)))

for i in range(len(angles_1)):
	a = int(angles_1[i]/angle_bin_size)
	polar_array_1[a][i] = all_cs_1[i]


polar_array_2 = np.zeros((int(360/angle_bin_size), len(angles_2)))

for i in range(len(angles_2)):
	a = int(angles_2[i]/angle_bin_size)
	polar_array_2[a][i] = all_cs_2[i]


polar_array_1[polar_array_1 == 0] = 'nan'
polar_vals_1 = np.nanmean(polar_array_1, axis=1)
polar_vals_1 = np.nan_to_num(polar_vals_1)
polar_vals_1 = np.append(polar_vals_1,polar_vals_1[0])


polar_array_2[polar_array_2 == 0] = 'nan'
polar_vals_2 = np.nanmean(polar_array_2, axis=1)
polar_vals_2 = np.nan_to_num(polar_vals_2)
polar_vals_2 = np.append(polar_vals_2,polar_vals_2[0])


fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
plt.polar(polar_axis,polar_vals_1,'b-',label='No Flow')
plt.polar(polar_axis,polar_vals_2,'r-',label='Flow')
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
plt.legend(loc=(1,1))
#plt.legend([no_flow, flow],["No Flow","Flow"])
plt.show()

# sns.displot(angles)
# plt.show()




