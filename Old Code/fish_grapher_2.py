import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera

n_fish = 5
b_parts = ["head","tail","midline2","midline1","midline3"]

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

	return(fish_dict,time_points)

def dict_to_fish_time(f_dict,fish_num,time):
	x = np.zeros(n_fish)
	y = np.zeros(n_fish)

	for i in range(n_fish):
		x[i] = f_dict[fish_num][b_parts[i]]["x"][time]
		y[i] = f_dict[fish_num][b_parts[i]]["y"][time]

	return(np.stack((x, y), axis=-1))



fish_dict,time_points = DLC_CSV_to_dict(num_fish = n_fish, fish_parts = b_parts)

fig, axes = plt.subplots(2)
camera = Camera(fig)

t = np.linspace(0, 2 * np.pi, 128, endpoint=False)

axes[0].axis([0, 2000, 0, 2000])
axes[1].axis([0, 2000, 0, 2000])

print(t)

for i in t:
    axes[0].plot(t, np.sin(t + i), color='blue')
    axes[1].plot(t, np.sin(t - i), color='blue')
    camera.snap()

animation = camera.animate()
animation.save('animation.mp4')



