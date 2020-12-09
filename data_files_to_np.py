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

# def angle_between(x1s, y1s, x2s, y2s):
#     ang1 = np.arctan2(x1s, y1s)
#     ang2 = np.arctan2(x2s, y2s)
#     print(np.rad2deg(ang1),np.rad2deg(ang2))
#     deg_diff = np.rad2deg((ang1 - ang2) % (2 * np.pi))
#     sys.exit()
#     return deg_diff

for file_name in os.listdir(data_folder):
	if file_name.endswith(".csv") and flow in file_name and dark in file_name and turb in file_name:
		num_data += 1
		data_files.append(file_name)

all_xs = []
all_ys = []
all_cs = []
all_hs = []

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

	fish_head_xs = []
	fish_head_ys = []

	fish_midline_1_xs = []
	fish_midline_1_ys = []

	for i in range(n_fish):
		fish_head_xs.append(fish_dict[i]["head"]["x"])
		fish_head_ys.append(fish_dict[i]["head"]["y"])

		fish_midline_1_xs.append(fish_dict[i]["midline1"]["x"])
		fish_midline_1_ys.append(fish_dict[i]["midline1"]["y"])

	fish_head_xs = np.asarray(fish_head_xs)
	fish_head_ys = np.asarray(fish_head_ys)

	fish_midline_1_xs = np.asarray(fish_midline_1_xs)
	fish_midline_1_ys = np.asarray(fish_midline_1_ys)

	#Go through all timepoints with each fish as the center one
	#Edited so that all the time points are done at once through the magic of numpy
	for f in range(n_fish):

		main_fish_x = fish_head_xs[f]
		main_fish_y = fish_head_ys[f]

		main_fish_m1_x = fish_midline_1_xs[f]
		main_fish_m1_y = fish_midline_1_ys[f]

		main_fish_heading = np.rad2deg(np.arctan2(main_fish_m1_y-main_fish_y,main_fish_m1_x-main_fish_x))

		#This prevents perfect symetry and doubling up on fish
		for g in range(f+1,n_fish):
			other_fish_x = fish_head_xs[g]
			other_fish_y = fish_head_ys[g]

			other_fish_m1_x = fish_midline_1_xs[g]
			other_fish_m1_y = fish_midline_1_ys[g]

			other_fish_heading = np.rad2deg(np.arctan2(other_fish_m1_y-other_fish_y,other_fish_m1_x-other_fish_x))

			#This order is so that the heatmap faces correctly upstream
			x_diff = (main_fish_x - other_fish_x)/cnvrt_pix_bl[f]
			y_diff = (other_fish_y - main_fish_y)/cnvrt_pix_bl[f]

			#This is to make it not go over and wrap around at the 180, -180 side
			angle_diff = 180-abs(180-abs(main_fish_heading - other_fish_heading))

			#-5.500432249997483 179.63034648078283

			for i in range(len(x_diff)):
				# print()
				# print(main_fish_x[i],main_fish_y[i],main_fish_m1_x[i],main_fish_m1_y[i])
				# #print(main_fish_m1_x[i]-main_fish_x[i],main_fish_m1_y[i]-main_fish_y[i])
				# print(other_fish_x[i],other_fish_y[i],other_fish_m1_x[i],other_fish_m1_y[i])
				# print(angle_diff[i],main_fish_heading[i],other_fish_heading[i])

				all_xs.append(abs(x_diff[i]))
				all_ys.append(y_diff[i])

				# all_xs.append(-1*x_diff)
				# all_ys.append(y_diff)

				# -1 * log(x+1)+1
				all_cs.append(-1*math.log(slope_array[f][j][i]+1)+1)
				#all_cs.append(-1*math.log(slope_array[f][j][i]+1)+1)

				all_hs.append(angle_diff[i])

all_xs = np.asarray(all_xs)
all_ys = np.asarray(all_ys)
all_cs = np.asarray(all_cs)
all_hs = np.asarray(all_hs)

with open(save_file, 'wb') as f:
	np.save(f, all_xs)
	np.save(f, all_ys)
	np.save(f, all_cs)
	np.save(f, all_hs)

	