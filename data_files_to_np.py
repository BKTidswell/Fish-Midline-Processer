import os, sys
import math
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Ellipse
from fish_core import *
from PIL import Image
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.signal import savgol_filter, resample

data_folder = os.getcwd()+"/Finished_Fish_Data/"
flows = ["F0","F2"]
darks = ["DN"]
turbs = ["TN"]

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

for flow in flows:
	for dark in darks:
		for turb in turbs:

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
					for j in range(i+1,n_fish):

						cvn_n = 15
						cvn_n_half = int(cvn_n/2)

						signal_1 = normalize_signal(fish_perp[i][:,5])
						signal_2 = normalize_signal(fish_perp[j][:,5])

						mv_avg_1 = moving_average(moving_average(signal_1,cvn_n),cvn_n)
						mv_avg_2 = moving_average(moving_average(signal_2,cvn_n),cvn_n)

						short_signal_1 = signal_1[cvn_n:-cvn_n+2]
						short_signal_2 = signal_2[cvn_n:-cvn_n+2]

						pn_signal_1 = short_signal_1-mv_avg_1
						pn_signal_2 = short_signal_2-mv_avg_2

						# doubled_signal_1 = np.append(resample(pn_signal_1,int(len(pn_signal_1)/2)),resample(pn_signal_1,int(len(pn_signal_1)/2)))
						# doubled_signal_2 = np.append(np.append(resample(pn_signal_1,int(len(pn_signal_1)/4)),resample(pn_signal_1,int(len(pn_signal_1)/4))),
						# 							 np.append(resample(pn_signal_1,int(len(pn_signal_1)/4)),resample(pn_signal_1,int(len(pn_signal_1)/4))))

						#Get the signal for each with Hilbert phase
						analytic_signal_1 = hilbert(pn_signal_1)
						instantaneous_phase_1 = np.unwrap(np.angle(analytic_signal_1))

						analytic_signal_2 = hilbert(pn_signal_2)
						instantaneous_phase_2 = np.unwrap(np.angle(analytic_signal_2))

						# #Now get the slope
						# dx = np.diff(instantaneous_phase_main)
						# dy = np.diff(instantaneous_phase)

						#This normalizes from 0 to 1. Not sure I should do this, but here we are
						#If I don't it really throws off the scale.

						#10/13 slope is now 0 when they are aligned and higher when worse. 

						#10/16 uses the get slope function for smoother slope
						#abs_diff = get_slope(instantaneous_phase_1,instantaneous_phase_2)
						#norm_slope = abs(slope-1)

						#12/14 I think that I just need to subtract actually. So 0 is best and > is worse still
						#https://math.stackexchange.com/questions/1000519/phase-shift-of-two-sine-curves/1000703#1000703

						# Actually I want the slope of the subtracted lines. Also this is a nightmare and I hate math
						# and curves and lines and smoothing. God I hope this works. 

						#Ok, so 12/15. How it works now is that the formulat is 2^-x and the *2 for sync_slope
						# makes it so that if one is twice the freq of another then it gets a value of 1, which 
						# then becomes 0.5 in the end. So 1x = 0, 2x = 0.5, 3x = 0.25, 4x = 0.125 etc.
						# It's not perfect certainly. The *2 is just what I picked that worked best when I doubled
						# it up. Also if one signal is 2x and the other is 4x, then then value difference is 2. 
						# So it's not perfect on doubling, but it is on the total times faster from base.
						# But why is the base the base?? Unclear to me at least. Still it works. 
						abs_diff_smooth = savgol_filter(abs(instantaneous_phase_2 - instantaneous_phase_1),11,1)
						sync_slope = abs(np.gradient(abs_diff_smooth))*2

						#stuff_agh = -1*np.log(sync_slope2+1)+1
						#stuff_agh = np.power(2,sync_slope*-1)

						# fig, axs = plt.subplots(7)
						# fig.suptitle('Vertically stacked subplots')
						# axs[0].plot(range(len(short_signal_1)), short_signal_1)
						# axs[0].plot(range(len(mv_avg_1)), mv_avg_1, "g")

						# axs[1].plot(range(len(short_signal_2)), short_signal_2,"r")
						# axs[1].plot(range(len(mv_avg_2)), mv_avg_2, "m")

						# axs[2].plot(range(len(pn_signal_1)), pn_signal_1)
						# axs[2].plot(range(len(pn_signal_2)), pn_signal_2,"r")

						# axs[3].plot(range(len(instantaneous_phase_1)), instantaneous_phase_1)
						# axs[3].plot(range(len(instantaneous_phase_2)), instantaneous_phase_2,"r")

						# axs[4].plot(range(len(abs_diff_smooth)), abs_diff_smooth)

						# axs[5].plot(range(len(sync_slope)), sync_slope)
						# axs[5].set_ylim(-0.1,2)

						# #axs[5].plot(range(len(sync_slope2)), sync_slope2,"r")

						# axs[6].plot(range(len(stuff_agh)), stuff_agh,"r")

						# plt.show()

						# sys.exit()

						#Now copy it all over. Time is reduced becuase diff makes it shorter
						for t in range(len(sync_slope)):
							slope_array[i][j][t] = sync_slope[t]

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

				fish_angles = np.zeros(0)
				fish_angles_2 = np.zeros(0)

				for f in range(n_fish):
					#This prevents perfect symetry and doubling up on fish
					main_fish_x = fish_head_xs[f]
					main_fish_y = fish_head_ys[f]

					main_fish_n_x = np.roll(main_fish_x, -1)
					main_fish_n_y = np.roll(main_fish_y, -1)

					#Get vectors for angle calculations
					mfish_vecx = main_fish_n_x - main_fish_x
					mfish_vecy = main_fish_n_y - main_fish_y

					mfish_angle = np.rad2deg(np.arctan2(mfish_vecy,mfish_vecx))
					#sns.distplot(mfish_angle)

					#fish_angles = np.append(fish_angles,mfish_angle)

					for g in range(f+1,n_fish):

						# n is for "next"
						# roll by 1 so the last pair value is not good, but that's why I use "range(len(x_diff)-1)" later
					
						other_fish_x = fish_head_xs[g]
						other_fish_y = fish_head_ys[g]

						other_fish_n_x = np.roll(other_fish_x, -1)
						other_fish_n_y = np.roll(other_fish_y, -1)

						ofish_vecx = other_fish_n_x - other_fish_x
						ofish_vecy = other_fish_n_y - other_fish_y

						ofish_angle = np.rad2deg(np.arctan2(ofish_vecy,ofish_vecx))

						#fish_angles_2 = np.append(fish_angles_2,ofish_angle)

						#This is to make it not go over and wrap around at the 180, -180 side
						#angle_diff = (mfish_vecx * ofish_vecx + mfish_vecy * ofish_vecy) / (np.sqrt(mfish_vecx**2 + mfish_vecy**2) * np.sqrt(ofish_vecx**2 + ofish_vecy**2))

						#This is to make it map from 0 to 1 to make subtracting easier
						#angle_diff = (angle_diff+1)/2

						#This makes it so that it only returns values from 0 to 180, and always gets the smallest distance 
						angle_diff = 180 - abs(180 - abs(mfish_angle-ofish_angle))

						#Then maps it so that 0 is worst and 1 is best
						angle_diff = 1-(angle_diff/180)

						fish_angles = np.append(fish_angles,angle_diff)

						#This order is so that the heatmap faces correctly upstream
						x_diff = (main_fish_x - other_fish_x)/cnvrt_pix_bl[f]
						y_diff = (other_fish_y - main_fish_y)/cnvrt_pix_bl[f]

						#This -1 is so that the last value pair (which is wrong bc of roll) is not counted.
						for i in range(len(x_diff)-1):
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

							#12/14
							#e^(-x/4)
							#all_cs.append(-1*math.log(slope_array[f][j][i]+1)+1)
							#all_cs.append(-1*math.log(slope_array[f][j][i]+1)+1)

							all_cs.append(np.power(2,slope_array[f][j][i]*-1))

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

			#sns.distplot(fish_angles)
			#sns.distplot(fish_angles_2)
			#plt.show()

	