import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

class AnimatedScatter(object):
	"""An animated scatter plot using matplotlib.animations.FuncAnimation."""
	def __init__(self, numpoints=1):
		self.numpoints = numpoints
		self.stream = self.data_stream()

		# Setup the figure and axes...
		self.fig, self.ax = plt.subplots(2)
 		# Then setup FuncAnimation.
		self.ani = animation.FuncAnimation(self.fig, *self.update, interval=5, 
										  init_func=self.setup_plot, blit=True)

	def setup_plot(self):
		"""Initial drawing of the scatter plot."""
		self.scats = [0,0,0,0,0]

		x, y = next(self.stream).T
		self.scats[0] = self.ax[0].scatter(x, y, s=5, vmin=0, vmax=1,
									cmap="jet", edgecolor="k")
		self.ax[0].axis([0, 2000, 0, 2000])
		# For FuncAnimation's sake, we need to return the artist we'll be using
		# Note that it expects a sequence of artists, thus the trailing comma.

		self.scats[1] = self.ax[1].scatter(x, y, s=5, vmin=0, vmax=1,
									cmap="jet", edgecolor="k")
		self.ax[1].axis([0, 2000, 0, 2000])

		return self.scats[0],self.scats[1],

	def data_stream(self):
		"""Generate a random walk (brownian motion). Data is scaled to produce
		a soft "flickering" effect."""
		for i in range(time_points):
			two_fish = np.concatenate((dict_to_fish_time(fish_dict,0,i),dict_to_fish_time(fish_dict,1,i)), axis = 0)
			#yield(dict_to_fish_time(fish_dict,0,i))
			yield two_fish

		plt.close()

	def update(self,i):
		"""Update the scatter plot."""
		data = next(self.stream)

		# Set x and y data...
		self.scats[0].set_offsets(data[:, :2])
		self.scats[1].set_offsets(data[:, :2])

		# We need to return the updated artist for FuncAnimation to draw..
		# Note that it expects a sequence of artists, thus the trailing comma.
		return self.scats[0],self.scats[1],


#for getting the straight line. Use the smoothed trajectory of the second body part (midline1)
#that gets direction of motion, make that the x axis

if __name__ == '__main__':
	a = AnimatedScatter()
	plt.show()