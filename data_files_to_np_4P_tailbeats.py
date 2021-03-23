import os, sys
import math
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Ellipse
from fish_core_4P import *
from PIL import Image
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.signal import savgol_filter, resample, find_peaks

data_folder = os.getcwd()+"/Finished_Fish_Data_4P/"
flows = ["F0","F2"]
darks = ["DN"]
turbs = ["TN"]

##DO MOVING AVERAGE OF DATA INSTEAD OF CHUNKS

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def mean_tailbeat_chunk(data,tailbeat_len):
	max_tb_frame = len(data)-len(data)%med_tailbeat_len
	mean_data = np.zeros(max_tb_frame)

	for k in range(max_tb_frame):
		start = k//med_tailbeat_len * med_tailbeat_len
		end = (k//med_tailbeat_len + 1) * med_tailbeat_len

		mean_data[k] = np.mean(data[start:end])

	return mean_data[::med_tailbeat_len]

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

				# 3/17
				#Okay so. Now the thing I need to do is not do each time point, but an average over each tailbeat
				# Since we don't want doubling frames to double the sample size
				# And I'm just..... not sure how to do that. tail beats can be different lengths! For each fish!
				# But the sync is between two of them! So which tailbeats do I take?
				# I am...not sure? Maybe I can find the average tailbeat length for one fish, and then the average
				# length between them? But it would be good to keep it constant over all trials
				# But I don't want it to change everytime I put in new data. Well let's just see what that average might be

				#First create an n_fish x n_fish x timepoints array to store the slopes in

				peak_width = 5
				tailbeat_lens = []

				for i in range(n_fish):
					cvn_n = 15
					signal_1 = normalize_signal(fish_perp[i][:,b_parts.index("tailtip")])
					mv_avg_1 = moving_average(moving_average(signal_1,cvn_n),cvn_n)
					short_signal_1 = signal_1[cvn_n:-cvn_n+2]
					pn_signal_1 = short_signal_1-mv_avg_1
					peaks_1, _ = find_peaks(pn_signal_1, width = peak_width)
					dist_between = np.diff(peaks_1)

					for d in dist_between:
						tailbeat_lens.append(d)


				tailbeat_lens = np.asarray(tailbeat_lens)

				# print(tailbeat_lens)
				# print(np.mean(tailbeat_lens))
				# print(np.median(tailbeat_lens))


				med_tailbeat_len = int(np.median(tailbeat_lens))

				#Get a reduced length number of points
				tailbeat_points = (time_points - time_points%med_tailbeat_len)//med_tailbeat_len

				slope_array = np.zeros((n_fish,n_fish,tailbeat_points))

				#print(slope_array.shape)

				for i in range(n_fish):
					for j in range(i+1,n_fish):

						#3/22 Adding in speed to see how that maps to tailbeats based on
						# the work done in Swain et al., 2015

						#End result is that it is very different and maybe also useful
						# but isn't a replacement for it 1 to 1

						# speed_1 = get_dist_np(fish_dict[i]["head"]["x"][:-1],
						# 					  fish_dict[i]["head"]["y"][:-1],
						# 					  fish_dict[i]["head"]["x"][1:],
						# 					  fish_dict[i]["head"]["y"][1:])

						# speed_2 = get_dist_np(fish_dict[j]["head"]["x"][:-1],
						# 					  fish_dict[j]["head"]["y"][:-1],
						# 					  fish_dict[j]["head"]["x"][1:],
						# 					  fish_dict[j]["head"]["y"][1:])

						#I don't like that this is such a magic number but here we are
						# It does smooth well though
						cvn_n = 15

						signal_1 = normalize_signal(fish_perp[i][:,b_parts.index("tailtip")])
						signal_2 = normalize_signal(fish_perp[j][:,b_parts.index("tailtip")])

						mv_avg_1 = moving_average(moving_average(signal_1,cvn_n),cvn_n)
						mv_avg_2 = moving_average(moving_average(signal_2,cvn_n),cvn_n)

						short_signal_1 = signal_1[cvn_n:-cvn_n+2]
						short_signal_2 = signal_2[cvn_n:-cvn_n+2]

						pn_signal_1 = short_signal_1-mv_avg_1
						pn_signal_2 = short_signal_2-mv_avg_2

						peaks_1, _ = find_peaks(pn_signal_1, width = peak_width)
						peaks_2, _ = find_peaks(pn_signal_2, width = peak_width)

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

						#So now we're going to find the mean sync over each of the tailbeat_len chunks
						#We find here the max frame we'll go to in sync slope to get even beats

						#Okay so now I'm trying it not in chunks but as a moving average
						# max_tb_frame = len(sync_slope)-len(sync_slope)%med_tailbeat_len

						# mean_sync_beats = np.zeros(max_tb_frame)
						# #print(max_tb_frame)

						# #This makes it go to an even number of TBs, no hanging edge
						# for k in range(max_tb_frame):
						# 	start = k//med_tailbeat_len * med_tailbeat_len
						# 	end = (k//med_tailbeat_len + 1) * med_tailbeat_len

						# 	mean_sync_beats[k] = np.mean(sync_slope[start:end])

						mean_sync_beats = mean_tailbeat_chunk(sync_slope,med_tailbeat_len)
						norm_sync = np.power(2,mean_sync_beats*-1)
						# #sync_no_avg = np.power(2,sync_slope*-1)

						#This is the code I use to graph things when things go wrong
						# Or when I need to show code and new processes to Eric
						# fig, axs = plt.subplots(7)
						# fig.suptitle('Vertically stacked subplots')
						# axs[0].plot(range(len(short_signal_1)), short_signal_1)
						# axs[0].plot(range(len(mv_avg_1)), mv_avg_1, "g")
						# #axs[0].plot(peaks_1, short_signal_1[peaks_1], "x")

						# axs[1].plot(range(len(short_signal_2)), short_signal_2,"r")
						# axs[1].plot(range(len(mv_avg_2)), mv_avg_2, "m")
						# #axs[1].plot(peaks_2, short_signal_2[peaks_2], "x")

						# axs[2].plot(range(len(pn_signal_1)), pn_signal_1)
						# axs[2].plot(range(len(pn_signal_2)), pn_signal_2,"r")
						# axs[2].plot(peaks_1, pn_signal_1[peaks_1], "x")
						# axs[2].plot(peaks_2, pn_signal_2[peaks_2], "x")

						# axs[3].plot(range(len(instantaneous_phase_1)), instantaneous_phase_1)
						# axs[3].plot(range(len(instantaneous_phase_2)), instantaneous_phase_2,"r")

						# axs[4].plot(range(len(abs_diff_smooth)), abs_diff_smooth)

						# axs[5].plot(range(len(sync_slope)), sync_slope)
						# axs[5].plot(range(len(mean_sync_beats)), mean_sync_beats)
						# axs[5].set_ylim(-0.1,2)

						# #axs[6].plot(range(len(sync_no_avg)), sync_no_avg,)
						# axs[6].plot(range(len(norm_sync)), norm_sync,)

						# if norm_sync[-1] == 0:
						# 	plt.show()

						# plt.close()
						#sys.exit()

						#norm_sync_out = norm_sync[::med_tailbeat_len]

						#Now copy it all over. Time is reduced becuase diff makes it shorter
						for t in range(len(norm_sync)):
							slope_array[i][j][t] = norm_sync[t]


				#sys.exit()
				#print(slope_array)
				#print(slope_array.shape)

				#Now all of these need to also be done in means by the tailbeat length of time

				fish_head_xs = []
				fish_head_ys = []

				fish_midline_1_xs = []
				fish_midline_1_ys = []

				for i in range(n_fish):
					fish_head_xs.append(fish_dict[i]["head"]["x"])
					fish_head_ys.append(fish_dict[i]["head"]["y"])

					fish_midline_1_xs.append(fish_dict[i]["midline2"]["x"])
					fish_midline_1_ys.append(fish_dict[i]["midline2"]["y"])

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

					#Then turn the x and y vector to get the angle
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

						#Then turn the x and y vector to get the angle
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

						#3/23 Here is where I should be taking tailbeat averages, not before
						x_diff = mean_tailbeat_chunk(x_diff,med_tailbeat_len)
						y_diff = mean_tailbeat_chunk(y_diff,med_tailbeat_len)
						angle_diff = mean_tailbeat_chunk(angle_diff,med_tailbeat_len)

						#3/22
						#So the norm_sync is 1 smaller than the xdiff-1 array so we're using it instead
						# this is becuase with all the smoothing and stuff more data gets clipped
						#I thought abotu using the slope array but I can't becasue it is to long
						# Because it is made with timepoints in length and so has trailing zeros
						#Someday I need to clean up this code and also use better iterators
						#Not today though!
						for i in range(len(norm_sync)):
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


							#3/22 This had been j instead of g for a long time.
							# This is a mess and idk how that may have messed things up
							# I guess we'll see what the data actually looks like!!!
							all_cs.append(slope_array[f][g][i])

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

			# sns.distplot(all_hs)
			# #sns.distplot(fish_angles_2)
			# plt.show()

	