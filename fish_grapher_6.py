
from fish_core import *

file_name = "2020_7_28_29_TN_DN_F2_V1DLC_resnet50_L8FVJul4shuffle1_100000_sk_filtered.csv"

#Create the fish dict and get the time points
fish_dict,time_points = DLC_CSV_to_dict(num_fish = n_fish, fish_parts = b_parts_csv, file = file_name)

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

#Plot the tail points
tt_fig, tt_axs = plt.subplots(n_fish)
tt_fig.suptitle('Hilbert Transform Phase Correlation')
time_x = np.linspace(0, time_points-2, time_points-1)

boxplot_data = []

for i in range(n_fish):
	analytic_signal_main = hilbert(normalize_signal(fish_perp[0][:,5]))
	instantaneous_phase_main = np.unwrap(np.angle(analytic_signal_main))

	analytic_signal = hilbert(normalize_signal(fish_perp[i][:,5]))
	instantaneous_phase = np.unwrap(np.angle(analytic_signal))

	slope = get_slope(instantaneous_phase_main,instantaneous_phase)
	norm_slope = abs(slope-1)

	if i > 0:
		boxplot_data.append(norm_slope)

	tt_axs[i].plot(time_x[2:-2],norm_slope,color = fish_colors[i])
	tt_axs[i].set_ylim(-1, 10)

plt.show()


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


dim = 2000
offset = dim/2

time_pos_array = np.zeros((dim,dim,time_points))
#time_pos_array[time_pos_array == 0] = np.NaN

fish_head_xs = []
fish_head_ys = []

for i in range(n_fish):
	fish_head_xs.append(fish_dict[i]["head"]["x"])
	fish_head_ys.append(fish_dict[i]["head"]["y"])

fish_head_xs = np.asarray(fish_head_xs)
fish_head_ys = np.asarray(fish_head_ys)

xs = []
ys = []
cs = []

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
			x_diff = round(main_fish_x - other_fish_x)
			y_diff = round(other_fish_y - main_fish_y)

			x_pos = int(x_diff+offset)
			y_pos = int(y_diff+offset)

			xs.append(x_diff)
			ys.append(y_diff)

			#time_pos_array[y_pos][x_pos][i] = slope_array[f][j][i]
			time_pos_array[y_pos][x_pos][i] = 1


fig, ax = plt.subplots()
counts, xedges, yedges, im = ax.hist2d(xs, ys, bins=75, cmap = "jet", cmin = 1)
ax.set_ylim(min(min(xedges),min(yedges)), max(max(xedges),max(yedges)))
ax.set_xlim(min(min(xedges),min(yedges)), max(max(xedges),max(yedges)))
im.set_clim(1,50)
fig.colorbar(im)
plt.show()

# fp = plt.scatter(xs, ys, s=10, c=cs, alpha=0.5, cmap='jet')
# plt.colorbar(fp)
# plt.show()

heatmap_array = np.sum(time_pos_array, axis=2)
#time_pos_array[time_pos_array == 0] = np.NaN
#heatmap_array = np.nanmean(time_pos_array, axis=2)

#heatmap_array = np.nan_to_num(heatmap_array)

#remove the center point for scaling:
heatmap_array[int(dim/2)][int(dim/2)] = 0

new_dim = 100
#shrunk_map = shrink_nanmean(heatmap_array,new_dim,new_dim)
shrunk_map = shrink_sum(heatmap_array,new_dim,new_dim)


shrunk_map[shrunk_map == 0] = np.NaN


# plt.imshow(shrunk_map, cmap='hot', interpolation='nearest')
# plt.set_ylim(0, 100)
# plt.show()

fig, ax = plt.subplots()
# ax.set_ylim(len(shrunk_map)/-2, len(shrunk_map)/2)
# ax.set_xlim(len(shrunk_map)/-2, len(shrunk_map)/2)
ax.set_ylim(0,new_dim-1)
im = ax.imshow(shrunk_map,cmap='jet')
im.set_clim(0,75)
fig.colorbar(im)
plt.show()

#plot_fish_vid(fish_dict,fish_para,fish_perp,fish_paths,time_points)