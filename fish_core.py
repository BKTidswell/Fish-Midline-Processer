import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
from scipy import interpolate
import numpy.ma as ma
from scipy.interpolate import splprep, splev
from matplotlib.gridspec import GridSpec
from scipy.signal import argrelextrema, correlate,hilbert

#Defines the number of fish and the body parts that are tracked by DLC
n_fish = 8
b_parts_csv = ["head","tailbase","midline2","midline1","midline3","tailtip"]
n_b_parts = len(b_parts_csv)
b_parts = ["head","midline1","midline2","midline3","tailbase","tailtip",]

#Sets the colors used later for the Matplotlib graphs
fish_colors = ["red","orange","yellow","green","blue","purple","pink","grey"]

def get_slope(x,y):
	slope_array = np.zeros(len(x))

	for i in range(2,len(x)-2):
		#This gets the slope from the points surrounding i so that the signal is less noisy
		slope = (((y[i+1]-y[i-1]) / (x[i+1]-x[i-1])) + ((y[i+2]-y[i-2]) / (x[i+2]-x[i-2]))) / 2
		slope_array[i] = slope

	return(slope_array[2:-2])

#normalizes between -1 and 1
def normalize_signal(data):
	min_val = np.min(data)
	max_val = np.max(data)

	divisor = max(max_val,abs(min_val))

	return data/divisor

def shrink_nanmean(data, rows, cols):
    new_data = data.reshape(rows, int(data.shape[0]/rows), cols, int(data.shape[1]/cols))
    nan_mean_data = np.nanmean(np.nanmean(new_data,axis=1),axis=2)
    #return new_data.sum(axis=1).sum(axis=2)
    return nan_mean_data

def shrink_sum(data, rows, cols):
    return data.reshape(rows, int(data.shape[0]/rows), cols, int(data.shape[1]/cols)).sum(axis=1).sum(axis=2)

#This function takes the CSV from DLC and outputs a dictionary with the data stored in a more managable way:
#The structure is:
#   fish_dict:
#       individual_fish:
#           x_cord for each frame
#           y for each frame
#           probability for each frame

def DLC_CSV_to_dict(num_fish,fish_parts,file):
	data_points = ["x","y","prob"]
	  
	fish_dict = {}

	for i in range(num_fish):
		fish_dict[i] = {}
		for part in fish_parts:
			fish_dict[i][part] = {}
			for point in data_points:
				fish_dict[i][part][point] = []
	  
	# To open Workbook 
	fish_data = pd.read_csv(file)

	cols = fish_data.columns
	time_points = len(fish_data[cols[0]])

	for i in range(0,len(cols)-1):
		#Every 3 [x,y,p] times number of body parts columns we move onto a new fish
		fish_num = math.floor(i/(n_b_parts*3))
		#Every 3 [x,y,p] columns we move to a new body part, and we rotate through all
		fish_part = fish_parts[int(math.floor(i/3)%n_b_parts)]
		#rotate through the 3 [x,y,p] data type columns
		data_point = data_points[i%3]
		#Store the column data in the dict
		#Ignore the first 3 rows as those are column labels
		fish_dict[fish_num][fish_part][data_point] = fish_data[cols[i+1]][3:time_points].astype(float).to_numpy()

	return(fish_dict,time_points-3)

#The point of the function here is to take the dict data and change it to a better format for matplotlib
#Returns a 2 x Frames array with x and y data
def dict_to_fish_time(f_dict,fish_num,time):
	x = np.zeros(n_b_parts)
	y = np.zeros(n_b_parts)

	#Goes through and fills in the x and y arrays

	#UH this usage of i here is very wrong and only worked before by coincidence
	for i in range(n_b_parts):
		x[i] = f_dict[fish_num][b_parts[i]]["x"][time]
		y[i] = f_dict[fish_num][b_parts[i]]["y"][time]

	return([x,y])

#So this whole function is used to get the predicted paths of any x and y points
#This can be used for getting the average midline, and can also be used to get the
#   points if the probablities for some are low to smooth those out.
#The smoothing value will need to be changed in that case
def splprep_predict(x,y,maxTime):

	#Create x,y, and time arrays
	x = np.asarray(x)
	y = np.asarray(y)
	t = np.asarray(range(maxTime))
	  
	#Remove missing data using a mask
	#A masked array allows the use of arrays with missing data  
	x = ma.masked_where(np.isnan(x), x)
	y = ma.masked_where(np.isnan(y), y)
	  
	if(np.any(x.mask)):
		x = x[~x.mask]
	  
	if(np.any(y.mask)):
		y = y[~y.mask]
	  
	#So this goes from the second data point to the end
	#And this function only appends the new data if there is a change between them
	#splprep does not work if the coordinates are the same one after another, and this removes
	# those points

	newX = [x[0]]
	newY = [y[0]]

	for i in range(min(len(x),len(y)))[1:]:
		if (abs(x[i] - x[i-1]) > 1e-4) or (abs(y[i] - y[i-1]) > 1e-4):
			newX.append(x[i])
			newY.append(y[i])

	newX = np.asarray(newX)
	newY = np.asarray(newY)

	#Runs splprep with s as smoothing to generate the splines
	tck, u = splprep([newX, newY], s=10**5)
	#Creates the time points to that they go from 0 to 1 with a number of 
	# devisions equal to the number of frames
	newU = np.arange(0,1,t[1]/(t[-1]+t[1]))
	#Runs splev to generate new points from the spline from 0 to 1
	# with a number ofdivisions equal to the number of frames
	new_points = splev(newU, tck)
	  
	return(new_points)

#Gets the distance between two points
def get_dist(p1,p2):
	return(math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)) )

#Uses dot and cross products to get parallel and perpendicular distance
def get_para_perp_dist(p_fish,p_predict,p_next):

	#Creates the position vector (from the point to the predicted point)
	#	and the swimming vector (from the predicted point to the next one)
	position_vector = np.asarray([p_predict[0]-p_fish[0],p_predict[1]-p_fish[1],0])
	swim_vector = np.asarray([p_predict[0]-p_next[0],p_predict[1]-p_next[1],0])

	#Finds the length of the swimming vector and converts it to a unit vector
	vecDist = math.sqrt(swim_vector[0]**2 + swim_vector[1]**2)
	swim_vector = swim_vector/vecDist

	#Creates the vector prependicular to the swimming vector
	perp_swim_vector = np.cross(swim_vector,[0,0,1])

	#Uses the dot product between the postion vector and the swimming vectors
	#	to get the parallel and perpendictuar coordinates
	para_coord = np.dot(position_vector,swim_vector)
	perp_coord = np.dot(position_vector,perp_swim_vector)

	return(para_coord,perp_coord)

#Thsi uses the functions above to generate the midline graph for one fish
def generate_midline(one_fish,time_points):

	#Use the midline 1 and midline 3 data a the line to track
	fish_m1_x = one_fish["midline1"]["x"]
	fish_m1_y = one_fish["midline1"]["y"]

	fish_m3_x = one_fish["midline3"]["x"]
	fish_m3_y = one_fish["midline3"]["y"]

	#Predict the path of that midline 1 point
	predict_fish_m1 = splprep_predict(fish_m1_x,fish_m1_y,time_points)
	predict_fish_m3 = splprep_predict(fish_m3_x,fish_m3_y,time_points)
	
	para_a = []
	perp_a = []

	body_line = np.zeros((time_points-1,2,2))

	#For each frame...
	for i in range(time_points-1):

		#Create a temporary array to store the para and 
		#	perp distances for each point on the body
		temp_para = np.zeros(n_b_parts)
		temp_perp = np.zeros(n_b_parts)

		#Then for each of the parts...
		for j in range(len(b_parts)):
			#Get the x and y coordinates...
			fish_x_b = one_fish[b_parts[j]]["x"]
			fish_y_b = one_fish[b_parts[j]]["y"]

			#Put them into x,y point formats
			current_fish = [ fish_x_b[i],fish_y_b[i] ]
			m1_point = [ predict_fish_m1[0][i],predict_fish_m1[1][i] ]
			m3_point = [ predict_fish_m3[0][i],predict_fish_m3[1][i] ]
			
			#And pass them through get_para_perp_dist to get the points out
			temp_para[j],temp_perp[j] = get_para_perp_dist(current_fish,m1_point,m3_point)

		#Append those to the larger array
		para_a.append(temp_para)
		perp_a.append(temp_perp)

		body_line[i][0][0] = fish_m1_x[i]
		body_line[i][0][1] = fish_m3_x[i]

		body_line[i][1][0] = fish_m1_y[i]
		body_line[i][1][1] = fish_m3_y[i]

	#And finally return the para and perp arrays, as well as the predicted midline path
	#Now does not include graphing midline path since it is the body line that is plotted
	return(para_a,perp_a,body_line)

#Plots an animation of the data with moving fish and midline centering
def plot_fish_vid(fish_dict,fish_para,fish_perp,fish_paths,time_points):
	#Set up the figure
	fig = plt.figure(figsize=(14,7))
	axes = []

	#There's probably a better way to do this, but this sets up the 8 subplots
	# 	And the one large one
	gs = GridSpec(3, 6, figure=fig, wspace = 0.5, hspace=0.5)
	axes.append( fig.add_subplot(gs[0:3, 0:3]) )
	axes.append( fig.add_subplot(gs[0, 3]))
	axes.append( fig.add_subplot(gs[0, 4]))
	axes.append( fig.add_subplot(gs[0, 5]))
	axes.append( fig.add_subplot(gs[1, 3]))
	axes.append( fig.add_subplot(gs[1, 5]))
	axes.append( fig.add_subplot(gs[2, 3]))
	axes.append( fig.add_subplot(gs[2, 4]))
	axes.append( fig.add_subplot(gs[2, 5]))  


	for i in range(8):
		axes[i+1].set(xlim=(-250, 250), ylim=(-250, 250))

	#Make an array to store the graphs in
	ims=[]

	for i in range(time_points-1):
		#Save the tempoary plots
		temp_plots = []

		for j in range(n_fish):
			#Get the data from the fish position
			fish = dict_to_fish_time(fish_dict,j,i)
			#Plot the data of that fish on the big plot
			#The trailing , is 100% needed for it to all work and be appended to the plot list
			bigplot, = axes[0].plot(fish[0], fish[1], color = fish_colors[j], marker='o')
			
			#Add the midline plots to the big graph
			#new_mid_plot, = axes[0].plot(fish_paths[j][i][0], fish_paths[j][i][1], color = "black")

			#Use the smaller plots and put each individual fish there using para and perp distances
			new_fish_plot, = axes[j+1].plot(fish_para[j][i], fish_perp[j][i], color = fish_colors[j], marker='o')

			#Append the big plot and smaller fish plots to the teporary arrays
			temp_plots.append(new_fish_plot)
			temp_plots.append(new_mid_plot)
			temp_plots.append(bigplot)

		#Append the temp plots to the ims in order to build up the stack of plots to animate
		ims.append(temp_plots)

	#Run through the ims to create the animation like a flipbook
	ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=4000)
	plt.show()