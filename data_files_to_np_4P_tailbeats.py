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

#Matplotlib breaks with Qt now in big sur :(
mpl.use('tkagg')

data_folder = os.getcwd()+"/Finished_Fish_Data_4P/"
flows = ["F0","F2"]
darks = ["DN"]
turbs = ["TN"]

##DO MOVING AVERAGE OF DATA INSTEAD OF CHUNKS

## Actually don't

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def mean_tailbeat_chunk(data,tailbeat_len):
	max_tb_frame = len(data)-len(data)%tailbeat_len
	mean_data = np.zeros(max_tb_frame)

	for k in range(max_tb_frame):
		start = k//tailbeat_len * tailbeat_len
		end = (k//tailbeat_len + 1) * tailbeat_len

		mean_data[k] = np.mean(data[start:end])

	return mean_data[::tailbeat_len]


def angular_mean_tailbeat_chunk(data,tailbeat_len):
	max_tb_frame = len(data)-len(data)%tailbeat_len
	mean_data = np.zeros(max_tb_frame)

	for k in range(max_tb_frame):
		start = k//tailbeat_len * tailbeat_len
		end = (k//tailbeat_len + 1) * tailbeat_len

		data_range = data[start:end]

		cos_mean = np.mean(np.cos(data_range))
		sin_mean = np.mean(np.sin(data_range))

		#SIN then COSINE
		angular_mean = np.rad2deg(np.arctan2(sin_mean,cos_mean))
		mean_data[k] = angular_mean

	return mean_data[::tailbeat_len]

def two_p_zero_y_intercept(x1,y1,x2,y2):
	m = (y2-y1)/(x2-x1)
	intercept = (-1*y1)/m + x1

	return intercept

for flow in flows:
	for dark in darks:
		for turb in turbs:

			save_file = "data_{}_{}_{}.npy".format(flow,dark,turb)

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
			all_hs_old = []
			all_tbf = []
			all_spd = []			

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

				median_pix_bl = np.median(cnvrt_pix_bl)

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

				# 6/16
				#Changed the tailbeat length calcualtion to use zero crossings instead of peaks
				# I also made sure that zero crossings are only going from negative to positive, and not both

				#First create an n_fish x n_fish x timepoints array to store the slopes in

				peak_width = 5
				tailbeat_lens = []

				out_tbf = []

				for i in range(n_fish):
					cvn_n = 15
					signal_1 = normalize_signal(fish_perp[i][:,b_parts.index("tailtip")])
					mv_avg_1 = moving_average(moving_average(signal_1,cvn_n),cvn_n)
					short_signal_1 = signal_1[cvn_n:-cvn_n+2]
					pn_signal_1 = short_signal_1-mv_avg_1
					#peaks_1, _ = find_peaks(pn_signal_1, width = peak_width)
					zero_crossings = np.where(np.diff(np.sign(pn_signal_1)) > 0)[0]
					dist_between = np.diff(zero_crossings)

					# fig, axs = plt.subplots(2)
					# fig.suptitle('Vertically stacked subplots')
					# axs[0].plot(range(len(short_signal_1)), short_signal_1)
					# axs[0].plot(range(len(mv_avg_1)), mv_avg_1, "g")

					# axs[1].plot(range(len(pn_signal_1)), pn_signal_1)
					# axs[1].plot(zero_crossings, pn_signal_1[zero_crossings], "x")

					out_tbf.append(dist_between)

					for d in dist_between:
						tailbeat_lens.append(d)

				tailbeat_lens = np.asarray(tailbeat_lens)

				# print(tailbeat_lens)
				# print(tailbeat_lens.shape)
				# print(np.mean(tailbeat_lens))
				# print(np.median(tailbeat_lens))


				med_tailbeat_len = int(np.median(tailbeat_lens))

				#Get a reduced length number of points
				tailbeat_points = (time_points - time_points%med_tailbeat_len)//med_tailbeat_len

				slope_array = np.zeros((n_fish,n_fish,tailbeat_points))
				tailbeat_freq_array = np.zeros((n_fish,n_fish,tailbeat_points))
				spd_diff_array = np.zeros((n_fish,n_fish,tailbeat_points))

				#print(slope_array.shape)

				for i in range(n_fish):
					for j in range(i+1,n_fish):

						#3/22 Adding in speed to see how that maps to tailbeats based on
						# the work done in Swain et al., 2015

						#End result is that it is very different and maybe also useful
						# but isn't a replacement for it 1 to 1

						#6/9: We're doing this again. Make it into BL/s as well
						#60 for 60 fps

						#6/16 Now doing it using the median BL for fish so that it isn't different for all fish
						#  and we're getting the relative speed between fish instead of their own speed
						# So it's a measure of how fast the distance between them is changing

						speed_1 = get_dist_np(fish_dict[i]["head"]["x"][:-1],
											  fish_dict[i]["head"]["y"][:-1],
											  fish_dict[i]["head"]["x"][1:],
											  fish_dict[i]["head"]["y"][1:])/median_pix_bl*60

						speed_2 = get_dist_np(fish_dict[j]["head"]["x"][:-1],
											  fish_dict[j]["head"]["y"][:-1],
											  fish_dict[j]["head"]["x"][1:],
											  fish_dict[j]["head"]["y"][1:])/median_pix_bl*60

						between_speed = np.diff(get_dist_np(fish_dict[i]["head"]["x"],
															fish_dict[i]["head"]["y"],
															fish_dict[j]["head"]["x"],
															fish_dict[j]["head"]["y"])/median_pix_bl)*60

						#I don't like that this is such a magic number but here we are
						# It does smooth well though
						cvn_n = 15

						# cut_data_1 = fish_perp[i][:,b_parts.index("tailtip")]
						# cut_data_2 = fish_perp[j][:,b_parts.index("tailtip")]

						# cut_data_1[50:100] = np.nan
						# cut_data_2[50:100] = np.nan

						signal_1 = normalize_signal(fish_perp[i][:,b_parts.index("tailtip")])
						signal_2 = normalize_signal(fish_perp[j][:,b_parts.index("tailtip")])

						# signal_1 = normalize_signal(cut_data_1)
						# signal_2 = normalize_signal(cut_data_2)

						mv_avg_1 = moving_average(moving_average(signal_1,cvn_n),cvn_n)
						mv_avg_2 = moving_average(moving_average(signal_2,cvn_n),cvn_n)

						short_signal_1 = signal_1[cvn_n:-cvn_n+2]
						short_signal_2 = signal_2[cvn_n:-cvn_n+2]

						pn_signal_1 = short_signal_1-mv_avg_1
						pn_signal_2 = short_signal_2-mv_avg_2

						peaks_1, _ = find_peaks(pn_signal_1, width = peak_width)
						peaks_2, _ = find_peaks(pn_signal_2, width = peak_width)

						pn_signal_1_trim = pn_signal_1[~np.isnan(pn_signal_1)]
						pn_signal_2_trim = pn_signal_2[~np.isnan(pn_signal_2)]

						#Get the signal for each with Hilbert phase
						analytic_signal_1 = hilbert(pn_signal_1_trim)
						instantaneous_phase_1 = np.unwrap(np.angle(analytic_signal_1))

						analytic_signal_2 = hilbert(pn_signal_2_trim)
						instantaneous_phase_2 = np.unwrap(np.angle(analytic_signal_2))

						#60 for 60 fps
						#Only for the second fish since these aren't compared
						instantaneous_freq_2 = (np.diff(instantaneous_phase_2) / (2.0*np.pi) * 60)
						mean_tailbeat_freq_2 = mean_tailbeat_chunk(instantaneous_freq_2,med_tailbeat_len)


						#6/9 Calculatng Speed Diffs
						#Make it bodylength/s by the average between their 
						#No abs() here so it doesn't cut at zero
						spd_diff = speed_1-speed_2
						mean_tailbeat_spd_diff = mean_tailbeat_chunk(spd_diff,med_tailbeat_len)

						# out_array_1 = np.empty(len(instantaneous_phase_1)+50)
						# out_array_1[:] = np.NaN
						# out_array_1[0:50] = instantaneous_phase_1[0:50]
						# out_array_1[100:] = instantaneous_phase_1[50:]
						# instantaneous_phase_1 = out_array_1


						# out_array_2 = np.empty(len(instantaneous_phase_2)+50)
						# out_array_2[:] = np.NaN
						# out_array_2[0:50] = instantaneous_phase_2[0:50]
						# out_array_2[100:] = instantaneous_phase_2[50:]
						# instantaneous_phase_2 = out_array_2

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
						# sync_slope = abs(np.gradient(abs_diff_smooth))*2


						#4/5 Okay not trying this without the abs
						sync_slope = np.gradient(abs_diff_smooth)*2

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

						#4/5 Okay so like with the angles we move the abs here, after I take the mean
						# This allows for values >0 so that the mean can *be* 0 when things are reveresed, so 
						# It doesn't all just move to the middle. 
						norm_sync = np.power(2,abs(mean_sync_beats)*-1)
						
						#sync_no_avg = np.power(2,abs(sync_slope)*-1)

						#6/11 Working to do tailbeat sync based on zero crossing difference instead of hilbert
						# so that I don't need totally continuous data for this.

						#6/16 I realized that I was doubling up on zero crossings, counting both over and under,
						#  which was throwing things off, now it seems to be correct and matches up better.
						# The two options are so that the most data is captured, so that we get the first crossing
						#either from positive to negative or negaive to positive

						if pn_signal_1[0] > 0:
							zero_crossings_1 = np.where(np.diff(np.sign(pn_signal_1)) < 0)[0]
							zero_crossings_2 = np.where(np.diff(np.sign(pn_signal_2)) < 0)[0]
						else:
							zero_crossings_1 = np.where(np.diff(np.sign(pn_signal_1)) > 0)[0]
							zero_crossings_2 = np.where(np.diff(np.sign(pn_signal_2)) > 0)[0]

						tailbeat_offsets = np.zeros((len(zero_crossings_1),len(zero_crossings_2)))
						tailbeat_offsets[:] = np.nan

						#Can't use i and j here!!!
						for k in range(len(zero_crossings_1)-1):
							next_point = np.where((zero_crossings_2 >= zero_crossings_1[k]) & (zero_crossings_2 < zero_crossings_1[k+1]))[0]
							for l in next_point:
								#So here I get the points and find when they actually cross to zero in terms of time, not frames
								# So get the time of each frame, and then find the intercept between the two points
								z_cross_time_1 = two_p_zero_y_intercept(zero_crossings_1[k]+1,
																		pn_signal_1[zero_crossings_1[k]+1],
																		zero_crossings_1[k],
																		pn_signal_1[zero_crossings_1[k]])/60

								z_cross_time_2 = two_p_zero_y_intercept(zero_crossings_2[l]+1,
																		pn_signal_2[zero_crossings_2[l]+1],
																		zero_crossings_2[l],
																		pn_signal_1[zero_crossings_2[l]])/60

								tailbeat_offsets[k][l] = z_cross_time_2 - z_cross_time_1

						#6/16 I am taking the absolute value here as a negative difference isn't important, and I would want the mean of a 
						# -3 and 3 offset here to be 3, not 0.
						tailbeat_means = abs(np.nanmean(tailbeat_offsets, axis=1))
						tailbeat_repeats = abs(np.diff(np.append(zero_crossings_1,len(pn_signal_1))))
						tailbeat_chunked_means = mean_tailbeat_chunk(np.repeat(tailbeat_means,tailbeat_repeats),med_tailbeat_len)

						#This is the code I use to graph things when things go wrong
						# Or when I need to show code and new processes to Eric
						# fig, axs = plt.subplots(8)
						# fig.suptitle('Vertically stacked subplots')
						# axs[0].plot(range(len(short_signal_1)), short_signal_1)
						# axs[0].plot(range(len(mv_avg_1)), mv_avg_1, "g")
						# #axs[0].plot(zero_crossings_1, short_signal_1[zero_crossings_1], "x")

						# axs[1].plot(range(len(short_signal_2)), short_signal_2,"r")
						# axs[1].plot(range(len(mv_avg_2)), mv_avg_2, "m")
						# #axs[1].plot(zero_crossings_1, short_signal_2[zero_crossings_1], "x")

						# axs[2].plot(range(len(pn_signal_1)), pn_signal_1)
						# axs[2].plot(range(len(pn_signal_2)), pn_signal_2,"r")
						# axs[2].plot(zero_crossings_1, pn_signal_1[zero_crossings_1], "x")
						# axs[2].plot(zero_crossings_2, pn_signal_2[zero_crossings_2], "x")
						# #axs[2].plot(peaks_1, pn_signal_1[peaks_1], "x")
						# #axs[2].plot(peaks_2, pn_signal_2[peaks_2], "x")

						# axs[3].plot(range(len(instantaneous_phase_1)), instantaneous_phase_1)
						# axs[3].plot(range(len(instantaneous_phase_2)), instantaneous_phase_2,"r")

						# axs[4].plot(range(len(abs_diff_smooth)), abs_diff_smooth)

						# axs[5].plot(range(len(sync_slope)), sync_slope)
						# axs[5].plot(range(len(mean_sync_beats)*med_tailbeat_len), np.repeat(mean_sync_beats,med_tailbeat_len))
						# axs[5].set_ylim(-2,2)

						# ##axs[6].plot(range(len(sync_no_avg)), sync_no_avg)
						# axs[6].plot(range(len(norm_sync)*med_tailbeat_len), np.repeat(norm_sync,med_tailbeat_len))

						# axs[7].plot(range(len(tailbeat_chunked_means)*med_tailbeat_len), np.repeat(tailbeat_chunked_means,med_tailbeat_len))
						# # axs[7].plot(range(len(spd_diff)), spd_diff)
						# # axs[7].plot(range(len(between_speed)), between_speed, "r")

						# plt.show()
						# #norm_sync_out = norm_sync[::med_tailbeat_len]

						# #Now copy it all over. Time is reduced becuase diff makes it shorter
						# sys.exit()

						for t in range(len(tailbeat_chunked_means)):
							slope_array[i][j][t] = tailbeat_chunked_means[t]
							tailbeat_freq_array[i][j][t] = mean_tailbeat_freq_2[t]
							spd_diff_array[i][j][t] = mean_tailbeat_spd_diff[t]

						# print(mean_tailbeat_freq_2)
						# sys.exit()


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

					#4/2 Addition. Should not be in the inner look as then it just gets
					# smaller and smaller over time
					mfish_angle = np.deg2rad(np.where(mfish_angle < 0, 360 - abs(mfish_angle), mfish_angle))

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

						#3/29 use dot product?
						ofish_angle = np.rad2deg(np.arctan2(ofish_vecy,ofish_vecx))

						#fish_angles_2 = np.append(fish_angles_2,ofish_angle)

						#This is to make it not go over and wrap around at the 180, -180 side
						#angle_diff = (mfish_vecx * ofish_vecx + mfish_vecy * ofish_vecy) / (np.sqrt(mfish_vecx**2 + mfish_vecy**2) * np.sqrt(ofish_vecx**2 + ofish_vecy**2))

						#This is to make it map from 0 to 1 to make subtracting easier
						#angle_diff = (angle_diff+1)/2

						#This makes it so that it only returns values from 0 to 180, and always gets the smallest distance 
						#angle_diff = 180 - abs(180 - abs(mfish_angle-ofish_angle))

						#Then maps it so that 0 is worst and 1 is best
						#3/30 commenting this out so it's just degrees off. 0 degrees of to 180
						#angle_diff = 1-(angle_diff/180)

						#4/2 Going off Eric's suggestions to have them both from 0 to 360 and then get the
						# angular mean. This means going from -180 to 180 and turning that to 0 to 360.
						# Where -179 is 181 and -1 is 359
						# Sooooooo
						# for negative numbers only:
						# 180 - abs(val) make -179 into 1 and -1 into 179
						# Then just add 180?
						# Okay so this is actually just 360 - abs(val)
						# and then back to rad

						ofish_angle = np.where(ofish_angle < 0, 360 - abs(ofish_angle), ofish_angle)


						angle_diff = mfish_angle-ofish_angle #np.arctan2(np.sin(mfish_angle-ofish_angle), np.cos(mfish_angle-ofish_angle))

						#And then we use the new angular_mean_tailbeat_chunk instead 

						#This order is so that the heatmap faces correctly upstream
						#6/16 Changed so that it uses the median body length of fish
						x_diff = (main_fish_x - other_fish_x)/median_pix_bl
						y_diff = (other_fish_y - main_fish_y)/median_pix_bl

						#This -1 is so that the last value pair (which is wrong bc of roll) is not counted.

						#old_angle_diff = np.mod(np.rad2deg(mfish_angle-ofish_angle),360)

						old_angle_diff = abs(np.rad2deg(np.arctan2(np.sin(mfish_angle-ofish_angle), np.cos(mfish_angle-ofish_angle))))

						#3/23 Here is where I should be taking tailbeat averages, not before
						x_diff = mean_tailbeat_chunk(x_diff,med_tailbeat_len)
						y_diff = mean_tailbeat_chunk(y_diff,med_tailbeat_len)

						#4/2 New angular_mean_tailbeat_chunk function 
						angle_diff = angular_mean_tailbeat_chunk(np.deg2rad(angle_diff),med_tailbeat_len)

						all_hs_old.extend(old_angle_diff)

						#3/30 In graphing this I see that arctan2 does mean there is bouncing between 180 and -180
						# However the 180 - abs(180 - abs(mfish_angle-ofish_angle)) makes it not matter as 180
						# and -180 giving a result of 0 degrees apart. Which is why I did that so long ago

						# print(np.rad2deg(mfish_angle)[med_tailbeat_len:med_tailbeat_len*2])
						# print(np.rad2deg(ofish_angle)[med_tailbeat_len:med_tailbeat_len*2])
						# print(old_angle_diff[med_tailbeat_len:med_tailbeat_len*2])
						# print(angle_diff[1])

						# fig, axs = plt.subplots(5)
						# fig.tight_layout()

						# axs[0].plot(np.rad2deg(mfish_angle))
						# axs[0].set_ylim(-20,380)
						# axs[0].set_xlabel("Frame #")
						# axs[0].set_ylabel("Heading Angle")
						# axs[0].title.set_text("Fish {f} Heading".format(f=f))

						# axs[1].plot(np.rad2deg(ofish_angle))
						# axs[1].set_ylim(-20,380)
						# axs[1].set_xlabel("Frame #")
						# axs[1].set_ylabel("Heading Angle")
						# axs[1].title.set_text("Fish {g} Heading".format(g=g))

						# axs[2].plot(old_angle_diff)
						# #axs[2].set_ylim(-20,200)
						# axs[2].set_xlabel("Frame #")
						# axs[2].set_ylabel("Heading Angle")
						# axs[2].title.set_text("Fish Heading Difference")

						# axs[2].plot(np.repeat(angle_diff,med_tailbeat_len))
						# # axs[3].set_ylim(-20,200)
						# # axs[3].set_xlabel("Tailbeat Bins in Frame #")
						# # axs[3].set_ylabel("Heading Angle")
						# # axs[3].title.set_text("Mean Over Tailbeats")
						
						# axs[3].hist(old_angle_diff)
						# #axs[4].set_xlim(-20,200)
						# axs[3].set_xlabel("Heading Bins")
						# axs[3].set_ylabel("Count")
						# axs[3].title.set_text("Histogram Over Frames")
						
						# axs[4].hist(angle_diff)
						# #axs[5].set_xlim(-20,200)
						# axs[4].set_xlabel("Heading Bins")
						# axs[4].set_ylabel("Count")
						# axs[4].title.set_text("Histogram Over Tailbeats")

						# plt.show()

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


							#5/27, so here the values are from -180 to 180, and that is the same as 180 to 180, with 0 and 360 ant 0
							#So I want it to be 0 to 360 so I can do circular stats on it
							# %360 works since a % 360 = a if a is positive
							# And for negatives -1 %360 = 359 and -180 % 360 = 180
							all_hs.append(angle_diff[i] % 360)

							all_tbf.append(tailbeat_freq_array[f][g][i])
							all_spd.append(spd_diff_array[f][g][i])


			all_xs = np.asarray(all_xs)
			all_ys = np.asarray(all_ys)
			all_cs = np.asarray(all_cs)
			all_hs = np.asarray(all_hs)
			all_tbf = np.asarray(all_tbf)
			all_hs_old = np.asarray(all_hs_old)

			# fig, axs = plt.subplots(2)
			# axs[0].hist(all_hs_old)
			# axs[1].hist(all_hs)	
			# plt.show() 

			with open(save_file, 'wb') as f:
				np.save(f, all_xs)
				np.save(f, all_ys)
				np.save(f, all_cs)
				np.save(f, all_hs)
				np.save(f, all_tbf)
				np.save(f, all_spd)

			# sns.distplot(all_hs)
			# #sns.distplot(fish_angles_2)
			# plt.show()

	