import os, sys
import math
import seaborn as sns
from scipy import stats
from fish_core_4P import *
from PIL import Image
import matplotlib.image as mpimg
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#Matplotlib breaks with Qt now in big sur :(
mpl.use('tkagg')

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
	all_tbf_1 = np.load(f_1)
	all_spd_1 = np.load(f_1)

with open(save_file_2, 'rb') as f_2:
	all_xs_2 = np.load(f_2)
	all_ys_2 = np.load(f_2)
	all_cs_2 = np.load(f_2)
	all_hs_2 = np.load(f_2)
	all_tbf_2 = np.load(f_2)
	all_spd_2 = np.load(f_2)


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
polar_axis = np.linspace(0,180,int(180/angle_bin_size)+1) - angle_bin_size/2
polar_axis = (polar_axis+angle_bin_size/2) * np.pi /180

all_dists_1 = get_dist_np(0,0,all_xs_1,all_ys_1)
all_dists_2 = get_dist_np(0,0,all_xs_2,all_ys_2)

dist_bin_size = 1
max_dist = 3
d_range = round_down(max(np.max(all_dists_1),np.max(all_dists_2)),base=dist_bin_size)
d_axis = np.linspace(0,max_dist,int(max_dist/dist_bin_size)+1)


polar_array_1 = np.zeros((int(180/angle_bin_size), max_dist, len(angles_1)))
polar_density_array_1 = np.zeros((int(180/angle_bin_size), max_dist, len(angles_1)))
polar_heading_array_1 = np.zeros((int(180/angle_bin_size), max_dist, len(angles_1)))
polar_tbf_array_1 = np.zeros((int(180/angle_bin_size), max_dist, len(angles_1)))
polar_spd_array_1 = np.zeros((int(180/angle_bin_size), max_dist, len(angles_1)))

#Saving data to CSV for r power analysis 
outStr = "{cond},{distBin},{angleBin},{dist},{angle},{heading},{coord},{tbf},{spd_diff},{angleBinSize},{distBinSize}\n"

f = open("data_power_analysis/r_power_data.csv", "w")
f.write(outStr.format(cond="cond",distBin="distBin",angleBin="angleBin",dist="dist_v",angle="angle_v",heading="heading",coord="coord",tbf="tbf",spd_diff="spd_diff",angleBinSize="angleBinSize",distBinSize="distBinSize"))

for i in range(len(angles_1)):
	a = int(angles_1[i]/angle_bin_size)
	r = int(round_down(all_dists_1[i],base=dist_bin_size)/dist_bin_size)

	if r < max_dist:

		polar_array_1[a][r][i] = all_cs_1[i]
		polar_density_array_1[a][r][i] = 1
		polar_heading_array_1[a][r][i] = all_hs_1[i]
		polar_tbf_array_1[a][r][i] = all_tbf_1[i]
		polar_spd_array_1[a][r][i] = all_spd_1[i]

	f.write(outStr.format(cond=flow_1,distBin=r,angleBin=a,dist=all_dists_1[i],angle=angles_1[i],heading=all_hs_1[i],coord=all_cs_1[i],tbf=all_tbf_1[i],spd_diff=all_spd_1[i],angleBinSize=angle_bin_size,distBinSize=dist_bin_size))



polar_array_2 = np.zeros((int(180/angle_bin_size), max_dist, len(angles_2)))
polar_density_array_2 = np.zeros((int(180/angle_bin_size), max_dist, len(angles_1)))
polar_heading_array_2 = np.zeros((int(180/angle_bin_size), max_dist, len(angles_1)))
polar_tbf_array_2 = np.zeros((int(180/angle_bin_size), max_dist, len(angles_1)))
polar_spd_array_2 = np.zeros((int(180/angle_bin_size), max_dist, len(angles_1)))

for i in range(len(angles_2)):
	a = int(angles_2[i]/angle_bin_size)
	r = int(round_down(all_dists_2[i],base=dist_bin_size)/dist_bin_size)

	if r < max_dist:
		polar_array_2[a][r][i] = all_cs_2[i]
		polar_density_array_2[a][r][i] = 1
		polar_heading_array_2[a][r][i] = all_hs_2[i]
		polar_tbf_array_2[a][r][i] = all_tbf_2[i]
		polar_spd_array_2[a][r][i] = all_spd_2[i]

	f.write(outStr.format(cond=flow_2,distBin=r,angleBin=a,dist=all_dists_2[i],angle=angles_2[i],heading=all_hs_2[i],coord=all_cs_2[i],tbf=all_tbf_2[i],spd_diff=all_spd_2[i],angleBinSize=angle_bin_size,distBinSize=dist_bin_size))

f.close()

#Looking at synchonzaion differences.
polar_array_1[polar_array_1 == 0] = 'nan'
polar_vals_1 = np.nanmean(polar_array_1, axis=2)
polar_vals_1 = np.append(polar_vals_1,polar_vals_1[0].reshape(1, max_dist),axis=0)

polar_array_2[polar_array_2 == 0] = 'nan'
polar_vals_2 = np.nanmean(polar_array_2, axis=2)
polar_vals_2 = np.append(polar_vals_2,polar_vals_2[0].reshape(1, max_dist),axis=0)

#Get SE of arrays
se_polar_array_1 = stats.sem(polar_array_1, axis=2, nan_policy = "omit")
se_polar_array_1 = np.nan_to_num(np.asarray(se_polar_array_1))
se_polar_array_1 = np.append(se_polar_array_1,se_polar_array_1[0].reshape(1, max_dist),axis=0)

se_polar_array_2 = stats.sem(polar_array_2, axis=2, nan_policy = "omit")
se_polar_array_2 = np.nan_to_num(np.asarray(se_polar_array_2))
se_polar_array_2 = np.append(se_polar_array_2,se_polar_array_2[0].reshape(1, max_dist),axis=0)

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
polar_density_1 = polar_density_1/np.sum(polar_density_1)*100
polar_density_1 = np.append(polar_density_1,polar_density_1[0].reshape(1, max_dist),axis=0)

polar_density_2 = np.sum(polar_density_array_2, axis=2)
polar_density_2 = polar_density_2/np.sum(polar_density_2)*100
polar_density_2 = np.append(polar_density_2,polar_density_2[0].reshape(1, max_dist),axis=0)



#get the mean headings in each area
#Fix the mean here to work for circ mean
#
circ_multi = 2

polar_heading_array_1[polar_heading_array_1 == 0] = 'nan'
polar_headings_1 = np.rad2deg(stats.circmean(np.deg2rad(polar_heading_array_1), axis=2, nan_policy = "omit"))
polar_headings_1 = np.append(polar_headings_1,polar_headings_1[0].reshape(1, max_dist),axis=0)
#Take it from 0 to 360 to 0 to 180
polar_headings_1 = np.where(polar_headings_1 > 180,  abs(polar_headings_1 - 360), polar_headings_1) 


polar_heading_array_2[polar_heading_array_2 == 0] = 'nan'
polar_headings_2 = np.rad2deg(stats.circmean(np.deg2rad(polar_heading_array_2), axis=2, nan_policy = "omit"))
polar_headings_2 = np.append(polar_headings_2,polar_headings_2[0].reshape(1, max_dist),axis=0)
#Take it from 0 to 360 to 0 to 180
polar_headings_2 = np.where(polar_headings_2 > 180,  abs(polar_headings_2 - 360), polar_headings_2) 

#Get SE of heading arrays
se_polar_headings_1 = np.rad2deg(stats.circstd(np.deg2rad(polar_heading_array_1), axis=2, nan_policy = "omit"))
se_polar_headings_1 = np.nan_to_num(np.asarray(se_polar_headings_1))
se_polar_headings_1 = np.append(se_polar_headings_1,se_polar_headings_1[0].reshape(1, max_dist),axis=0)

se_polar_headings_2 = np.rad2deg(stats.circstd(np.deg2rad(polar_heading_array_2), axis=2, nan_policy = "omit"))
se_polar_headings_2 = np.nan_to_num(np.asarray(se_polar_headings_2))
se_polar_headings_2 = np.append(se_polar_headings_2,se_polar_headings_2[0].reshape(1, max_dist),axis=0)

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
density_diff = polar_density_1 - polar_density_2

data = [polar_vals_diff,sig_diff_array,polar_vals_1,polar_density_1,polar_vals_2,polar_density_2,polar_headings_1,polar_headings_2,heading_vals_diff,sig_diff_headings,density_diff]
names = [flow_1+"_"+flow_2+"_sync_diff.png",flow_1+"_"+flow_2+"_sync_sig_diff.png",flow_1+"_sync.png",flow_1+"_density.png",flow_2+"_sync.png",flow_2+"_density.png",flow_1+"_headings.png",flow_2+"_headings.png",flow_1+"_"+flow_2+"_heading_diff.png",flow_1+"_"+flow_2+"_heading_sig_diff.png",flow_1+"_"+flow_2+"_density_diff.png"]
titles = ["No Flow - Flow Synchronization", "No Flow - Flow Synchronization","No Flow Synchronization","No Flow Density","Flow Synchronization","Flow Density","No Flow Headings","Flow Headings","No Flow - Flow Headings", "No Flow - Flow Headings", "No Flow - Flow density"]
color = ["bwr","bwr","GnBu","GnBu","GnBu","GnBu","RdYlGn","RdYlGn","bwr","bwr","bwr"]
vmins = [-0.25,-1,0.75,0,0.75,0,0,0,-180,-1,-10]
vmaxs = [0.25,1,1,12,1,12,180,180,180,1,12]


x_data = np.asarray([polar_density_1.flatten(), polar_density_2.flatten()]).flatten()
y_data = np.asarray([polar_headings_1.flatten(), polar_headings_2.flatten()]).flatten()
error = np.asarray([se_polar_headings_1.flatten(), se_polar_headings_2.flatten()]).flatten()
colors = np.repeat([0.9,0.1],max_dist*7)


for i in range(len(data)):
	print(names[i])

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='polar')
	plt.pcolormesh(th, r, data[i], cmap = color[i], vmin=vmins[i], vmax=vmaxs[i])
	plt.title(titles[i],pad = -40)
	plt.xlabel("Distance (BL)",labelpad = -40)

	arr_png = mpimg.imread('fish_V.png')
	imagebox = OffsetImage(arr_png, zoom = 0.65)
	ab = AnnotationBbox(imagebox, (0, 0), frameon = False)
	ax.add_artist(ab)

	ax.set_xticks(polar_axis)
	ax.set_yticks(d_axis)

	if i in [3,5]:
		ax.set_ylim(0,3)

	ax.set_theta_zero_location("W")
	ax.set_theta_direction(-1)
	ax.set_thetamin(0)
	ax.set_thetamax(180)

	plt.plot(polar_axis, r, ls='none', color = 'k') 
	plt.grid()

	if i not in [1,9]: 
		plt.colorbar(pad = 0.1, shrink = 0.65)
	#plt.show()

	plt.savefig("Heatmaps_4P_tailbeats/"+names[i])
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




