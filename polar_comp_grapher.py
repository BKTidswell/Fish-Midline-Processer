import os
import math
import seaborn as sns
from scipy import stats
from fish_core import *
from PIL import Image
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

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
	all_hs_1 = np.load(f_1)

with open(save_file_2, 'rb') as f_2:
	all_xs_2 = np.load(f_2)
	all_ys_2 = np.load(f_2)
	all_cs_2 = np.load(f_2)
	all_hs_2 = np.load(f_2)

sns.distplot(all_hs_1,kde = False)
plt.show()

sns.distplot(all_hs_2,kde = False)
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
polar_axis = np.linspace(0,360,int(360/angle_bin_size)+1) - angle_bin_size/2
polar_axis = (polar_axis+angle_bin_size/2) * np.pi /180

all_dists_1 = get_dist_np(0,0,all_xs_1,all_ys_1)
all_dists_2 = get_dist_np(0,0,all_xs_2,all_ys_2)

dist_bin_size = 1
d_range = round_down(max(np.max(all_dists_1),np.max(all_dists_2)),base=dist_bin_size)
d_axis = np.linspace(0,d_range,int(d_range/dist_bin_size)+1)


polar_array_1 = np.zeros((int(360/angle_bin_size), len(d_axis), len(angles_1)))
polar_density_array_1 = np.zeros((int(360/angle_bin_size), len(d_axis), len(angles_1)))
polar_heading_array_1 = np.zeros((int(360/angle_bin_size), len(d_axis), len(angles_1)))

for i in range(len(angles_1)):
	a = int(angles_1[i]/angle_bin_size)
	r = int(round_down(all_dists_1[i],base=dist_bin_size)/dist_bin_size)

	polar_array_1[a][r][i] = all_cs_1[i]
	polar_density_array_1[a][r][i] = 1
	polar_heading_array_1[a][r][i] = all_hs_1[i]


polar_array_2 = np.zeros((int(360/angle_bin_size), len(d_axis), len(angles_2)))
polar_density_array_2 = np.zeros((int(360/angle_bin_size), len(d_axis), len(angles_1)))
polar_heading_array_2 = np.zeros((int(360/angle_bin_size), len(d_axis), len(angles_1)))

for i in range(len(angles_2)):
	a = int(angles_2[i]/angle_bin_size)
	r = int(round_down(all_dists_2[i],base=dist_bin_size)/dist_bin_size)

	polar_array_2[a][r][i] = all_cs_2[i]
	polar_density_array_2[a][r][i] = 1
	polar_heading_array_2[a][r][i] = all_hs_2[i]

#Looking at synchonzaion differences.
polar_array_1[polar_array_1 == 0] = 'nan'
polar_vals_1 = np.nanmean(polar_array_1, axis=2)
polar_vals_1 = np.append(polar_vals_1,polar_vals_1[0].reshape(1, (len(d_axis))),axis=0)

polar_array_2[polar_array_2 == 0] = 'nan'
polar_vals_2 = np.nanmean(polar_array_2, axis=2)
polar_vals_2 = np.append(polar_vals_2,polar_vals_2[0].reshape(1, (len(d_axis))),axis=0)

#Get SE of arrays
se_polar_array_1 = stats.sem(polar_array_1, axis=2, nan_policy = "omit")
se_polar_array_1 = np.nan_to_num(np.asarray(se_polar_array_1))
se_polar_array_1 = np.append(se_polar_array_1,se_polar_array_1[0].reshape(1, (len(d_axis))),axis=0)

se_polar_array_2 = stats.sem(polar_array_2, axis=2, nan_policy = "omit")
se_polar_array_2 = np.nan_to_num(np.asarray(se_polar_array_2))
se_polar_array_2 = np.append(se_polar_array_2,se_polar_array_2[0].reshape(1, (len(d_axis))),axis=0)

#See if these are all that different
polar_mean_diff_array = abs(polar_vals_1 - polar_vals_2)
polar_comp_error_array = se_polar_array_1 + se_polar_array_2

#See if the total difference is les than combined error
diff_array = polar_mean_diff_array > polar_comp_error_array
pos_neg_diff_array = np.where(polar_vals_1 - polar_vals_2 < 0, -1, 1)

sig_diff_array = pos_neg_diff_array*diff_array
sig_diff_array = sig_diff_array.astype('float')



#Makes data for density plots
polar_density_1 = np.sum(polar_density_array_1, axis=2)
polar_density_1 = polar_density_1/np.amax(polar_density_1)
polar_density_1 = np.append(polar_density_1,polar_density_1[0].reshape(1, (len(d_axis))),axis=0)

polar_density_2 = np.sum(polar_density_array_2, axis=2)
polar_density_2 = polar_density_2/np.amax(polar_density_2)
polar_density_2 = np.append(polar_density_2,polar_density_2[0].reshape(1, (len(d_axis))),axis=0)



#get the mean headings in each area
polar_heading_array_1[polar_heading_array_1 == 0] = 'nan'
polar_headings_1 = np.nanmean(polar_heading_array_1, axis=2)
polar_headings_1 = np.append(polar_headings_1,polar_headings_1[0].reshape(1, (len(d_axis))),axis=0)

polar_heading_array_2[polar_heading_array_2 == 0] = 'nan'
polar_headings_2 = np.nanmean(polar_heading_array_2, axis=2)
polar_headings_2 = np.append(polar_headings_2,polar_headings_2[0].reshape(1, (len(d_axis))),axis=0)

#Get SE of heading arrays
se_polar_headings_1 = stats.sem(polar_heading_array_1, axis=2, nan_policy = "omit")
se_polar_headings_1 = np.nan_to_num(np.asarray(se_polar_headings_1))
se_polar_headings_1 = np.append(se_polar_headings_1,se_polar_headings_1[0].reshape(1, (len(d_axis))),axis=0)

se_polar_headings_2 = stats.sem(polar_heading_array_2, axis=2, nan_policy = "omit")
se_polar_headings_2 = np.nan_to_num(np.asarray(se_polar_headings_2))
se_polar_headings_2 = np.append(se_polar_headings_2,se_polar_headings_2[0].reshape(1, (len(d_axis))),axis=0)

#See if these are all that different
#Reveresed from the other since higher is worse
polar_mean_diff_headings = abs(polar_headings_2 - polar_headings_1)
polar_comp_error_headings = se_polar_headings_1 + se_polar_headings_2

#See if the total difference is less than combined error
diff_headings = polar_mean_diff_headings > polar_comp_error_headings
pos_neg_diff_headings = np.where(polar_headings_2 - polar_headings_1 < 0, -1, 1)

sig_diff_headings = pos_neg_diff_headings*diff_headings
sig_diff_headings = sig_diff_headings.astype('float')

#sig_diff_array[sig_diff_array == 0] = 'nan'

#print(sig_diff_array)

r, th = np.meshgrid(d_axis, polar_axis)

polar_vals_diff = polar_vals_1 - polar_vals_2
heading_vals_diff = polar_headings_2 - polar_headings_1

data = [polar_vals_diff,sig_diff_array,polar_vals_1,polar_density_1,polar_vals_2,polar_density_2,polar_headings_1,polar_headings_2,heading_vals_diff,sig_diff_headings]
names = [flow_1+"_"+flow_2+"_diff.png",flow_1+"_"+flow_2+"_sig_diff.png",flow_1+"_sync.png",flow_1+"_density.png",flow_2+"_sync.png",flow_2+"_density.png",flow_1+"_headings.png",flow_2+"_headings.png",flow_1+"_"+flow_2+"_heading_diff.png",flow_1+"_"+flow_2+"_heading_sig_diff.png"]
titles = ["No Flow - Flow Synchronization", "No Flow - Flow Synchronization","No Flow Synchronization","No Flow Density","Flow Synchronization","Flow Density","No Flow Headings","Flow Headings","No Flow - Flow Headings", "No Flow - Flow Headings"]
color = ["bwr","bwr","GnBu","GnBu","GnBu","GnBu","GnBu","GnBu","bwr","bwr"]
vmins = [-1,-1,-1,0,-1,0,0,0,-1,-1]
vmaxs = [1,1,1,1,1,1,1,1,1,1]

for i in range(len(data)):
	print(names[i])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='polar')
	plt.pcolormesh(th, r, data[i], cmap = color[i], vmin=vmins[i], vmax=vmaxs[i])
	plt.title(titles[i],pad = -40)

	arr_png = mpimg.imread('fish.png')
	imagebox = OffsetImage(arr_png, zoom = 0.55)
	ab = AnnotationBbox(imagebox, (0, 0), frameon = False)
	ax.add_artist(ab)

	ax.set_xticks(polar_axis)
	ax.set_yticks(d_axis)
	ax.set_theta_zero_location("W")
	ax.set_theta_direction(-1)
	ax.set_thetamin(0)
	ax.set_thetamax(180)

	plt.plot(polar_axis, r, ls='none', color = 'k') 
	plt.grid()
	plt.colorbar(pad = 0.1, shrink = 0.65)
	#plt.show()

	plt.savefig("Heatmaps/"+names[i])
	plt.close()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='polar')
# plt.polar(polar_axis,polar_vals_1,'b-',label='No Flow')
# plt.polar(polar_axis,polar_vals_2,'r-',label='Flow')
# ax.set_theta_zero_location("N")
# ax.set_theta_direction(-1)
# plt.legend(loc=(1,1))
# #plt.legend([no_flow, flow],["No Flow","Flow"])
# plt.show()

# sns.displot(angles)




