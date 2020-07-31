import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

n_fish = 5
b_parts = ["head","tail","midline2","midline1","midline3"]
fish_colors = ["red","orange","green","blue","purple"]

def DLC_CSV_to_dict(num_fish,fish_parts):
	data_points = ["x","y","prob"]
	  
	fish_dict = {}

	for i in range(num_fish):
		fish_dict[i] = {}
		for part in fish_parts:
			fish_dict[i][part] = {}
			for point in data_points:
				fish_dict[i][part][point] = []

	# Give the location of the file 
	file = "N_LLine_A_1_TrimmedDLC_resnet50_Multi_VidsJun3shuffle1_3000_sk_filtered.csv"
	  
	# To open Workbook 
	fish_data = pd.read_csv(file)

	cols = fish_data.columns
	time_points = len(fish_data[cols[0]])

	for i in range(0,len(cols)-1):
		fish_num = math.floor(i/15)
		fish_part = fish_parts[math.floor(i/3)%5]
		data_point = data_points[i%3]

		fish_dict[fish_num][fish_part][data_point] = fish_data[cols[i+1]][3:time_points].astype(float).to_numpy()

	return(fish_dict,time_points-3)

def dict_to_fish_time(f_dict,fish_num,time):
	x = np.zeros(n_fish)
	y = np.zeros(n_fish)

	for i in range(n_fish):
		x[i] = f_dict[fish_num][b_parts[i]]["x"][time]
		y[i] = f_dict[fish_num][b_parts[i]]["y"][time]

	return([x,y])


fish_dict,time_points = DLC_CSV_to_dict(num_fish = n_fish, fish_parts = b_parts)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.axis([0, 2000, 0, 2000])
# ims=[]

fig, ax = plt.subplots(2,3)

for i in range(n_fish):
	print(i,math.floor(i/3),i%3)
	ax[math.floor(i/3),i%3].axis([0, 2000, 0, 2000])

ims=[]

for i in range(time_points):
	temp_plots = []

	for j in range(n_fish):
		fish = dict_to_fish_time(fish_dict,j,i)
		temp_plots.append(ax[math.floor(j/3),j%3].scatter(fish[0], fish[1], s=1, color= fish_colors[j]))

	ims.append(temp_plots)

ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,repeat_delay=2000)

plt.show()

