#This is in effect the same code as before, but now in 3D!

#This is the new code that will process all the CSVs of points into twos CSVs for R
# One will have fish summary stats, while the other will have the fish comparison values
#For the fish one the columns will be:
# Year, Month, Day, Trial, Abalation, Darkness, Flow, Fish, Tailbeat Num, Heading, Speed, TB Frequency

# For the between fish comparisons the columns will be: 
# Year, Month, Day, Trial, Abalation, Darkness, Flow, Fishes, Tailbeat Num, X Distance, Y Distance, Distance, Angle, Heading Diff, Speed Diff, Synchonization

#This is a lot of columns. But now instead of having multiple .npy files this will create an object for each of the positional data
# CSVs and then add them all together in the end. This will ideally make things easier to graph for testing, and not require so many 
# nested for loops. Fish may be their own objects inside of the trial objects so that they can be quickly compared. Which may mean that I need to 
# take apart fish_core_4P.py. In the end I think a lot of this will be easier to do with pandas and objects instead of reading line by line.

from scipy.signal import hilbert, savgol_filter, medfilt
from scipy.spatial import ConvexHull

from scipy.spatial import distance_matrix
import networkx as nx
import pylab

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec 
import pandas as pd
import numpy as np
import random
import math
import os, sys

#Matplotlib breaks with Qt now in big sur :(
mpl.use('tkagg')

fps = 60

#The moving average window is more of a guess tbh
moving_average_n = 35

#Tailbeat len is the median of all frame distances between tailbeats
tailbeat_len = 19


#Fish len is the median of all fish lengths in pixels
#Scale is different becasue of calibration
fish_len = 0.07862

#Used to try and remove weird times where fish extend
# Fish SD
fish_sd = 0.39930

# Fish Len Max?
fish_max_len = fish_len + 3*fish_sd

#Header list for reading the raw location CSVs
header = list(range(4))

def get_dist_np_2D(x1s,y1s,x2s,y2s):
    dist = np.sqrt((x1s-x2s)**2+(y1s-y2s)**2)
    return dist

def get_dist_np_3D(x1s,y1s,z1s,x2s,y2s,z2s):
    dist = np.sqrt((x1s-x2s)**2+(y1s-y2s)**2+(z1s-z2s)**2)
    return dist

def get_fish_length(fish):
    return (get_dist_np_3D(fish.head_x,fish.head_y,fish.head_z,fish.midline_x,fish.midline_y,fish.midline_z) + 
            get_dist_np_3D(fish.midline_x,fish.midline_y,fish.midline_z,fish.tailbase_x,fish.tailbase_y,fish.tailbase_z) +
            get_dist_np_3D(fish.tailbase_x,fish.tailbase_y,fish.tailbase_z,fish.tailtip_x,fish.tailtip_y,fish.tailtip_z))

def moving_average(x, w):
    #Here I am using rolling instead of convolve in order to not have massive gaps from a single nan
    return  pd.Series(x).rolling(window=w, min_periods=1).mean()

def normalize_signal(data):
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)

    divisor = max(max_val,abs(min_val))

    return data/divisor

def mean_tailbeat_chunk(data,tailbeat_len):
    max_tb_frame = len(data)-len(data)%tailbeat_len
    mean_data = np.zeros(max_tb_frame)

    for k in range(max_tb_frame):
        start = k//tailbeat_len * tailbeat_len
        end = (k//tailbeat_len + 1) * tailbeat_len

        mean_data[k] = np.mean(data[start:end])

    return mean_data[::tailbeat_len]

def median_tailbeat_chunk(data,tailbeat_len):
    max_tb_frame = len(data)-len(data)%tailbeat_len
    mean_data = np.zeros(max_tb_frame)

    for k in range(max_tb_frame):
        start = k//tailbeat_len * tailbeat_len
        end = (k//tailbeat_len + 1) * tailbeat_len

        mean_data[k] = np.median(data[start:end])

    return mean_data[::tailbeat_len]

def angular_mean_tailbeat_chunk(data,tailbeat_len):
    data = np.deg2rad(data)

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

def mean_tailbeat_chunk_sync(data,tailbeat_len):
    max_tb_frame = len(data)-len(data)%tailbeat_len
    mean_data = np.zeros(max_tb_frame)

    for k in range(max_tb_frame):
        start = k//tailbeat_len * tailbeat_len
        end = (k//tailbeat_len + 1) * tailbeat_len

        mean_data[k] = np.mean(data[start:end])

    return np.power(2,abs(mean_data[::tailbeat_len])*-1)

def x_intercept(x1,y1,x2,y2):
    m = (y2-y1)/(x2-x1)
    intercept = (-1*y1)/m + x1

    return intercept

#Calculates the uniformity of a distribution of phases or angles as a dimensionless number from 0 to 1
#Data must be given already normalized between 0 and 2pi

def rayleigh_cor(data):
    #Make an empy array the length of long axis of data
    out_cor = np.zeros(data.shape[0])

    #Get each time point as an array
    for i,d in enumerate(data):
        #Calcualte the x and y coordinates on the unit circle from the angle
        xs = np.cos(d)
        ys = np.sin(d)

        #Take the mean of x and of y
        mean_xs = np.nanmean(xs)
        mean_ys = np.nanmean(ys)

        #Find the magnitude of this new vector
        magnitude = np.sqrt(mean_xs**2 + mean_ys**2)

        out_cor[i] = magnitude

    return out_cor

class fish_data:
    def __init__(self, name, data, scorer, flow):
        #This sets up all of the datapoints that I will need from this fish
        self.name = name
        self.head_x = data[scorer][name]["head"]["x"].to_numpy() 
        self.head_y = data[scorer][name]["head"]["y"].to_numpy() 
        self.head_z = data[scorer][name]["head"]["z"].to_numpy() 

        self.midline_x = data[scorer][name]["midline2"]["x"].to_numpy() 
        self.midline_y = data[scorer][name]["midline2"]["y"].to_numpy() 
        self.midline_z = data[scorer][name]["midline2"]["z"].to_numpy() 

        self.tailbase_x = data[scorer][name]["tailbase"]["x"].to_numpy() 
        self.tailbase_y = data[scorer][name]["tailbase"]["y"].to_numpy() 
        self.tailbase_z = data[scorer][name]["tailbase"]["z"].to_numpy()

        self.tailtip_x = data[scorer][name]["tailtip"]["x"].to_numpy() 
        self.tailtip_y = data[scorer][name]["tailtip"]["y"].to_numpy()
        self.tailtip_z = data[scorer][name]["tailtip"]["z"].to_numpy()

        self.vec_x = []
        self.vec_y = []
        self.vec_z = []
        self.vec_xy = []

        self.tailtip_perp = [] 

        self.flow = flow
        
        #These are all blank and will be used for graphing
        self.normalized_tailtip = []
        self.tailtip_moving_average = []
        self.tailtip_zero_centered = []

        #These are the summary stats for the fish 
        self.yaw_heading = []
        self.pitch_heading = []
        self.speed = []
        self.zero_crossings = []
        self.tb_freq_reps = []
        self.body_lengths = []


        #Okay, so I want to remove data where fish are too long
        # So I am going to just do that, and void it out here with nans
        # Since all further functions draw from these positional values, I just null them here
        self.get_fish_BL()
        self.remove_long_fish()

        #This calcualtes the summary stats
        self.calc_yaw_heading()
        self.calc_pitch_heading()
        self.calc_speed()
        self.calc_tailtip_perp()
        self.calc_tb_freq()
        
        

    def get_fish_BL(self):
        self.body_lengths = (get_dist_np_3D(self.head_x,self.head_y,self.head_z,self.midline_x,self.midline_y,self.midline_z) + 
                             get_dist_np_3D(self.midline_x,self.midline_y,self.midline_z,self.tailbase_x,self.tailbase_y,self.tailbase_z) +
                             get_dist_np_3D(self.tailbase_x,self.tailbase_y,self.tailbase_z,self.tailtip_x,self.tailtip_y,self.tailtip_x))

    #Replaces the positional data with an NA if the fish is too long at that point in time
    #Trying to remove some of the weirdness from calibration
    def remove_long_fish(self):
        self.head_x[self.body_lengths > fish_max_len] = np.nan
        self.head_y[self.body_lengths > fish_max_len] = np.nan
        self.head_z[self.body_lengths > fish_max_len] = np.nan

        self.midline_x[self.body_lengths > fish_max_len] = np.nan
        self.midline_y[self.body_lengths > fish_max_len] = np.nan
        self.midline_z[self.body_lengths > fish_max_len] = np.nan

        self.tailbase_x[self.body_lengths > fish_max_len] = np.nan
        self.tailbase_y[self.body_lengths > fish_max_len] = np.nan
        self.tailbase_z[self.body_lengths > fish_max_len] = np.nan

        self.tailtip_x[self.body_lengths > fish_max_len] = np.nan
        self.tailtip_y[self.body_lengths > fish_max_len] = np.nan
        self.tailtip_z[self.body_lengths > fish_max_len] = np.nan

    #This function calcualtes the yaw heading of the fish at each timepoint
    #We are using body heading now, so midline to head, not head to next head
    def calc_yaw_heading(self):
        #Then we create a vector of the head minus the midline 
        self.vec_x = self.head_x - self.midline_x
        self.vec_y = self.head_y - self.midline_y

        #Then we use arctan to calculate the heading based on the x and y point vectors
        #Becasue of roll we don't want to the last value since it will be wrong
        self.yaw_heading = np.rad2deg(np.arctan2(self.vec_y,self.vec_x))

        # print(self.vec_x)
        # print(self.vec_y)
        # print(self.yaw_heading)

        # sys.exit()

    #This function calcualtes the pitch heading of the fish at each timepoint
    #We are using body heading now, so midline to head, not head to next head
    def calc_pitch_heading(self):
        #Then we create a vector of the head minus the midline 
        self.vec_x = self.head_x - self.midline_x
        self.vec_y = self.head_y - self.midline_y
        self.vec_z = self.head_z - self.midline_z

        self.vec_xy = get_dist_np_2D(0,0,self.vec_x,self.vec_y)

        #Then we use arctan to calculate the heading based on the x and y point vectors
        #Becasue of roll we don't want to the last value since it will be wrong
        self.pitch_heading = np.rad2deg(np.arctan2(self.vec_xy,self.vec_z))

    def calc_speed(self):
        #First we get the next points on the fish
        head_x_next = np.roll(self.head_x, -1)
        head_y_next = np.roll(self.head_y, -1)
        head_z_next = np.roll(self.head_z, -1)

        #Then we create a vector of the future point minus the last one
        speed_vec_x = head_x_next - self.head_x
        speed_vec_y = head_y_next - self.head_y
        speed_vec_z = head_z_next - self.head_z

        #Then we add the flow to the x value
        #Since (0,0) is in the upper left a positive vec_x value value means it is moving downstream
        #so I should subtract the flow value 
        #The flow value is mutliplied by the fish length since the vec_x values are in pixels, but it is in BLS so divide by fps
        vec_x_flow = speed_vec_x - (self.flow*fish_len)/fps

        #It is divided in order to get it in body lengths and then times fps to get BL/s
        self.speed = np.sqrt(vec_x_flow**2+speed_vec_y**2+speed_vec_z**2)[:-1]/fish_len * fps

    def calc_tailtip_perp(self):

        #First get the total number of frames
        total_frames = len(self.head_x)

        out_tailtip_perps = []

        #My old code does this frame by frame. There may be a way to vectorize it, but I'm not sure about that yet
        for i in range(total_frames):
            #Create a vector from the head to the tailtip and from the head to the midline
            tailtip_vec = np.asarray([self.head_x[i]-self.tailtip_x[i],self.head_y[i]-self.tailtip_y[i],self.head_z[i]-self.tailtip_z[i]])
            midline_vec = np.asarray([self.head_x[i]-self.midline_x[i],self.head_y[i]-self.midline_y[i],self.head_z[i]-self.midline_z[i]])

            #Then we make the midline vector a unit vector
            vecDist = np.sqrt(midline_vec[0]**2 + midline_vec[1]**2 + midline_vec[2]**2)
            midline_unit_vec = midline_vec/vecDist

            #We take the cross product of the midline unit vecotr to get a vector perpendicular to it
            perp_midline_vector = np.cross(midline_unit_vec,[0,0,1])

            #Finally, we calcualte the dot product between the vector perpendicular to midline vector and the 
            # vector from the head to the tailtip in order to find the perpendicular distance from the midline
            # to the tailtip
            out_tailtip_perps.append(np.dot(tailtip_vec,perp_midline_vector))

        self.tailtip_perp = out_tailtip_perps

    def calc_tb_freq(self):
        #First we normalize the tailtip from 0 to 1
        self.normalized_tailtip = normalize_signal(self.tailtip_perp)

        #then we take the moving average of this to get a baseline
        self.tailtip_moving_average = moving_average(self.normalized_tailtip,moving_average_n)

        #Next we zero center the tailtip path by subtracting the moving average
        self.tailtip_zero_centered = self.normalized_tailtip-self.tailtip_moving_average

        #Then we calculate where the signal crosses zero
        self.zero_crossings = np.where(np.diff(np.sign(self.tailtip_zero_centered)) > 0)[0]

        #Then we turn the distance between zero crossings to be taileat frequency 
        tb_freq = 60/np.diff(self.zero_crossings)

        #Then we repeat it to match the length of the tailbeats
        tailbeat_lengths = np.diff(np.append(self.zero_crossings,len(self.tailtip_zero_centered)))
        #This is so that if the first one does start at zero, a bit gets added on so that they all end up the same length
        #Some fish are gone entirely, and this breaks if I call an index when the fish don't have any points
        if(len(self.zero_crossings) > 0):
            tailbeat_lengths[0] += self.zero_crossings[0]

            #Sometimes there are no zero crossings, which seems wrong but I'm going to catching for it anyways rn
            if len(tailbeat_lengths) == 1:
                self.tb_freq_reps = np.repeat(0,tailbeat_lengths)
            else:
                #Now we append this here so that they all stay the same length. This basically extends the last tailbeat.
                self.tb_freq_reps = np.repeat(np.append(tb_freq,tb_freq[-1]),tailbeat_lengths) #[:len(tb_freq)])

    #Thsi function allows me to graph values for any fish without trying to cram it into a for loop somewhere
    def graph_values(self):
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(ncols = 3, nrows = 2) 

        ax0 = plt.subplot(gs[:,0])
        ax0.plot(self.head_x, self.head_y)
        ax0.scatter(self.head_x[0], self.head_y[0])
        ax0.plot(self.tailtip_x, self.tailtip_y)
        ax0.scatter(self.tailtip_x[0], self.tailtip_y[0])
        ax0.set_title("Fish Path (Blue = Head, Orange = Tailtip)")

        ax1 = plt.subplot(gs[0,1])
        ax1.plot(range(len(self.normalized_tailtip)), self.normalized_tailtip)
        ax1.plot(range(len(self.tailtip_moving_average)), self.tailtip_moving_average)
        ax1.set_title("Tailtip Perpendicular Distance")

        ax2 = plt.subplot(gs[1,1])
        ax2.plot(range(len(self.tailtip_zero_centered)), self.tailtip_zero_centered)
        ax2.plot(self.zero_crossings, self.tailtip_zero_centered[self.zero_crossings], "x")
        ax2.set_title("Tailtip Zero Crossings")

        ax3 = plt.subplot(gs[0,2])
        ax3.plot(range(len(self.speed)), self.speed)
        ax3.set_title("Speed")

        ax4 = plt.subplot(gs[1,2])
        ax4.plot(range(len(self.heading)), self.heading)
        ax4.set_title("Heading")

        plt.show()

class fish_comp:
    def __init__(self, fish1, fish2):
        self.name = fish1.name + "x" + fish2.name
        self.f1 = fish1
        self.f2 = fish2

        self.x_diff = []
        self.y_diff = []
        self.z_diff = []
        self.dist = []
        self.angle = []
        self.yaw_heading_diff = []
        self.pitch_heading_diff = []
        self.speed_diff = []
        self.tailbeat_offset_reps = []

        self.calc_dist()
        self.calc_angle()
        self.calc_yaw_heading_diff()
        self.calc_pitch_heading_diff()
        self.calc_speed_diff()
        self.calc_rayleigh_r()

        #self.graph_values()

    def calc_dist(self):        
        #Divided to get it into bodylengths
        self.x_diff = (self.f1.head_x - self.f2.head_x)/fish_len
        #the y_diff is negated so it faces correctly upstream
        self.y_diff = -1*(self.f1.head_y - self.f2.head_y)/fish_len
        self.z_diff = (self.f1.head_z - self.f2.head_z)/fish_len

        self.dist = get_dist_np_3D(0,0,0,self.x_diff,self.y_diff,self.z_diff)

    def calc_angle(self):
        #Calculate the angle of the x and y difference in degrees
        angle_diff = np.rad2deg(np.arctan2(self.y_diff,self.x_diff))
        #This makes it from 0 to 360
        #angle_diff_360 = np.mod(abs(angle_diff-360),360)
        #This rotates it so that 0 is at the top and 180 is below the fish for a sideways swimming fish model
        #self.angle = np.mod(angle_diff_360+90,360)

        #12/1/21: Back to making change notes. Now keeping it as the raw -180 to 180
        self.angle = angle_diff

    #Now with a dot product!
    def calc_yaw_heading_diff(self):
        f1_vector = np.asarray([self.f1.vec_x,self.f1.vec_y]).transpose()
        #print(f1_vector)
        f2_vector = np.asarray([self.f2.vec_x,self.f2.vec_y]).transpose()

        self.yaw_heading_diff = np.zeros(len(self.f1.vec_x))

        for i in range(len(self.f1.vec_x)):
            dot_product = np.dot(f1_vector[i], f2_vector[i])

            prod_of_norms = np.linalg.norm(f1_vector[i]) * np.linalg.norm(f2_vector[i])
            self.yaw_heading_diff[i] = np.degrees(np.arccos(dot_product / prod_of_norms))

    #Now with a dot product!
    def calc_pitch_heading_diff(self):
        f1_vector = np.asarray([self.f1.vec_xy,self.f1.vec_z]).transpose()
        #print(f1_vector)
        f2_vector = np.asarray([self.f2.vec_xy,self.f2.vec_z]).transpose()

        self.pitch_heading_diff = np.zeros(len(self.f1.vec_xy))

        for i in range(len(self.f1.vec_x)):
            dot_product = np.dot(f1_vector[i], f2_vector[i])

            prod_of_norms = np.linalg.norm(f1_vector[i]) * np.linalg.norm(f2_vector[i])
            self.pitch_heading_diff[i] = np.degrees(np.arccos(dot_product / prod_of_norms))

    def calc_heading_diff_filtered(self):
        #Makes sure that head wiggle doesn't mess up polarization
        f1_heading_og = self.f1.heading
        f2_heading_og = self.f2.heading

        self.f1.heading = savgol_filter(self.f1.heading,tailbeat_len,1)
        self.f2.heading = savgol_filter(self.f2.heading,tailbeat_len,1)

        self.heading_diff = np.rad2deg(np.arctan2(np.sin(np.deg2rad(self.f1.heading-self.f2.heading)),
                                                  np.cos(np.deg2rad(self.f1.heading-self.f2.heading))))

        for i in range(len(self.f1.heading)):
            print(f1_heading_og[i],f2_heading_og[i],self.f1.heading[i],self.f2.heading[i],self.heading_diff[i])

    def calc_speed_diff(self):
        self.speed_diff = self.f1.speed - self.f2.speed

    def calc_tailbeat_offset(self):
        #Setup an array to hold all the zero crossing differences
        tailbeat_offsets = np.zeros((len(self.f1.zero_crossings),len(self.f2.zero_crossings)))
        tailbeat_offsets[:] = np.nan

        for i in range(len(self.f1.zero_crossings)-2):
            #First we find all the points between each of the fish1 zero crossings
            next_point = np.where((self.f2.zero_crossings >= self.f1.zero_crossings[i]) & (self.f2.zero_crossings < self.f1.zero_crossings[i+1]))[0]

            #Then for each point we find the time intercept be calculating the x intercept
            # You find the slope between the point before and after the zero crossing, and get the
            # intercept from there.
            for j in next_point:
                f1_zero_cross_time = x_intercept(self.f1.zero_crossings[i]+1,
                                                 self.f1.tailtip_zero_centered[self.f1.zero_crossings[i]+1],
                                                 self.f1.zero_crossings[i],
                                                 self.f1.tailtip_zero_centered[self.f1.zero_crossings[i]])

                f2_zero_cross_time = x_intercept(self.f2.zero_crossings[j]+1,
                                                 self.f2.tailtip_zero_centered[self.f2.zero_crossings[j]+1],
                                                 self.f2.zero_crossings[j],
                                                 self.f2.tailtip_zero_centered[self.f2.zero_crossings[j]])

                #The Fish 2 value will be large so we substract Fish 1 from it to make it positive
                tailbeat_offsets[i][j] = f2_zero_cross_time - f1_zero_cross_time
                
        #Then we take the mean in case there are multiple Fish 2 tailbeats within 1 fish 1 tailbeat
        #We then take the difference between them since what we care about is the change in offset over time
        # specifically the absolute difference
        tailbeat_means = abs(np.diff(np.nanmean(tailbeat_offsets, axis=1)))
        #This gets the length of each tailbeat and then repeats it each time
        tailbeat_lengths = abs(np.diff(np.append(self.f1.zero_crossings,len(self.f1.tailtip_zero_centered))))

        #So now we have the average difference tailbeat onset time
        # And we divide by tailbeat length to see out of phase they are
        self.tailbeat_offset_reps = np.repeat(tailbeat_means,tailbeat_lengths[:len(tailbeat_means)])/tailbeat_len

    def calc_tailbeat_hilbert(self):

        #First we remove the NAs from the arrays
        f1_tailtip_na_rm = self.f1.tailtip_zero_centered[~np.isnan(self.f1.tailtip_zero_centered)]
        f2_tailtip_na_rm = self.f2.tailtip_zero_centered[~np.isnan(self.f2.tailtip_zero_centered)]

        #Then we get the hilbert signal and phase for both fish
        f1_analytic_signal = hilbert(f1_tailtip_na_rm)
        f1_instantaneous_phase = np.unwrap(np.angle(f1_analytic_signal))

        f2_analytic_signal = hilbert(f2_tailtip_na_rm)
        f2_instantaneous_phase = np.unwrap(np.angle(f2_analytic_signal))

        #Then we put the NAs back into the arrays in the right places
        f1_instantaneous_phase_nan = np.zeros(self.f1.tailtip_zero_centered.shape)
        f1_instantaneous_phase_nan[f1_instantaneous_phase_nan == 0] = np.nan
        f1_instantaneous_phase_nan[~np.isnan(self.f1.tailtip_zero_centered)] = f1_instantaneous_phase

        f2_instantaneous_phase_nan = np.zeros(self.f2.tailtip_zero_centered.shape)
        f2_instantaneous_phase_nan[f2_instantaneous_phase_nan == 0] = np.nan
        f2_instantaneous_phase_nan[~np.isnan(self.f2.tailtip_zero_centered)] = f2_instantaneous_phase

        #We smooth the signal and take the absolute values
        abs_diff_smooth = savgol_filter(abs(f2_instantaneous_phase_nan - f1_instantaneous_phase_nan),11,1)

        #Then we find the slope of the signal
        sync_slope = np.gradient(abs_diff_smooth)

        #We would do this, but instead we do it after averaging instead since the abs and the pwoer messes that up 
        #norm_sync = np.power(2,abs(sync_slope)*-1)

        self.tailbeat_offset_reps = sync_slope

    def calc_rayleigh_r(self):
        #We do the same first thing to calculate the phase offsets

        #Setup an array to hold all the zero crossing differences
        tailbeat_offsets = np.zeros((len(self.f1.zero_crossings),len(self.f2.zero_crossings)))
        tailbeat_offsets[:] = np.nan

        for i in range(len(self.f1.zero_crossings)-2):
            #First we find all the points between each of the fish1 zero crossings
            next_point = np.where((self.f2.zero_crossings >= self.f1.zero_crossings[i]) & (self.f2.zero_crossings < self.f1.zero_crossings[i+1]))[0]

            #Then for each point we find the time intercept be calculating the x intercept
            # You find the slope between the point before and after the zero crossing, and get the
            # intercept from there.
            for j in next_point:
                f1_zero_cross_time = x_intercept(self.f1.zero_crossings[i]+1,
                                                 self.f1.tailtip_zero_centered[self.f1.zero_crossings[i]+1],
                                                 self.f1.zero_crossings[i],
                                                 self.f1.tailtip_zero_centered[self.f1.zero_crossings[i]])

                f2_zero_cross_time = x_intercept(self.f2.zero_crossings[j]+1,
                                                 self.f2.tailtip_zero_centered[self.f2.zero_crossings[j]+1],
                                                 self.f2.zero_crossings[j],
                                                 self.f2.tailtip_zero_centered[self.f2.zero_crossings[j]])

                #The Fish 2 value will be larger so we substract Fish 1 from it to make it positive
                tailbeat_offsets[i][j] = f2_zero_cross_time - f1_zero_cross_time

        #Then we take the mean in case there are multiple Fish 2 tailbeats within 1 fish 1 tailbeat
        tailbeat_offset_means = np.nanmean(tailbeat_offsets, axis=1)

        #Now take these offset means and turn them into an angle from 0 to 2pi
        #Divide by the tailbeat length they were in and then multiply by 2pi
        tailbeat_offset_circ_scaled = (tailbeat_offset_means/tailbeat_len)*2*np.pi

        #Now stepping through by 5 we see calculate Rayleigh's R
        window = 5
        rayleigh_out = []

        for i in range(len(tailbeat_offset_circ_scaled)-window):

            #Get the window
            rayleigh_window = tailbeat_offset_circ_scaled[i:i+window]

            #Calcualte the x and y coordinates on the unit circle from the angle
            xs = np.cos(rayleigh_window)
            ys = np.sin(rayleigh_window)

            #Take the mean of x and of y
            mean_xs = np.nanmean(xs)
            mean_ys = np.nanmean(ys)

            #Find the magnitude of this new vector
            magnitude = np.sqrt(mean_xs**2 + mean_ys**2)

            #Append this value
            rayleigh_out.append(magnitude)

        tailbeat_lengths = abs(np.diff(np.append(self.f1.zero_crossings,len(self.f1.tailtip_zero_centered))))

        self.tailbeat_offset_reps = np.repeat(rayleigh_out,tailbeat_lengths[:len(rayleigh_out)])



    def graph_values(self):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(ncols = 5, nrows = 3) 

        ax0 = plt.subplot(gs[:,0])

        ax0.scatter(self.f1.head_x, self.f1.head_y, c = np.linspace(0,1,num = len(self.f1.head_x)), s = 2)
        ax0.scatter(self.f2.head_x, self.f2.head_y, c = np.linspace(0,1,num = len(self.f2.head_x)), s = 2)
        ax0.set_title("Fish Path (Blue = Fish 1, Orange = Fish 2)")

        ax1 = plt.subplot(gs[0,1])
        ax1.plot(range(len(self.dist)), self.dist)
        ax1.set_title("Distance")

        ax2 = plt.subplot(gs[1,1])
        ax2.plot(range(len(self.angle)), self.angle)
        ax2.set_title("Angle")

        ax3 = plt.subplot(gs[0,2])
        ax3.plot(range(len(self.f1.speed)), self.f1.speed)
        ax3.set_title("Fish 1 Speed")

        ax4 = plt.subplot(gs[1,2])
        ax4.plot(range(len(self.f2.speed)), self.f2.speed)
        ax4.set_title("Fish 2 Speed")

        ax5 = plt.subplot(gs[2,2])
        ax5.plot(range(len(self.speed_diff)), self.speed_diff)
        ax5.set_title("Speed Difference")

        ax6 = plt.subplot(gs[0,3])
        ax6.plot(range(len(self.f1.yaw_heading)), self.f1.yaw_heading)
        ax6.set_title("Fish 1 Heading")

        ax7 = plt.subplot(gs[1,3])
        ax7.plot(range(len(self.f2.yaw_heading)), self.f2.yaw_heading)
        ax7.set_title("Fish 2 Heading")

        ax8 = plt.subplot(gs[2,3])
        ax8.plot(range(len(self.yaw_heading_diff)), self.yaw_heading_diff)
        ax8.set_title("Heading Difference")

        ax9 = plt.subplot(gs[0,4])
        ax9.plot(range(len(self.f1.tailtip_zero_centered)), self.f1.tailtip_zero_centered)
        ax9.plot(self.f1.zero_crossings, self.f1.tailtip_zero_centered[self.f1.zero_crossings], "x")
        ax9.set_title("Fish 1 Tailbeats")

        ax10 = plt.subplot(gs[1,4])
        ax10.plot(range(len(self.f2.tailtip_zero_centered)), self.f2.tailtip_zero_centered)
        ax10.plot(self.f2.zero_crossings, self.f2.tailtip_zero_centered[self.f2.zero_crossings], "x")
        ax10.set_title("Fish 2 Tailbeats")

        ax11 = plt.subplot(gs[2,4])
        ax11.plot(range(len(self.tailbeat_offset_reps)), self.tailbeat_offset_reps)
        ax11.set_title("Tailbeat Offsets")

        plt.show()



class school_comps:
    def __init__(self, fishes, n_fish, flow):
        self.fishes = fishes
        self.n_fish = n_fish
        self.flow = flow

        self.school_center_x = []
        self.school_center_y = []
        self.school_center_z = []
        self.school_x_sd = []
        self.school_y_sd = []
        self.school_z_sd = []

        self.group_speed = []
        self.group_tb_freq =[]
        self.polarization = []

        self.correlation_strength = []
        self.nearest_neighbor_distance = []
        self.group_tailbeat_cor = []

        self.school_areas = []
        self.school_groups = []

        self.calc_school_pos_stats()

        self.remove_and_smooth_points()

        self.calc_school_speed()
        self.calc_school_tb_freq()
        self.calc_school_polarization()
        self.calc_nnd()
        self.calc_tailbeat_cor()
        self.calc_school_area()
        self.calc_school_groups()

        #self.graph_values()

    def calc_school_pos_stats(self):
        school_xs = [fish.head_x for fish in self.fishes]
        school_ys = [fish.head_y for fish in self.fishes]
        school_zs = [fish.head_z for fish in self.fishes]

        self.school_center_x = np.nanmean(school_xs, axis=0)
        self.school_center_y = np.nanmean(school_ys, axis=0)
        self.school_center_z = np.nanmean(school_zs, axis=0)

        self.school_x_sd = np.nanstd(school_xs, axis=0) / fish_len
        self.school_y_sd = np.nanstd(school_ys, axis=0) / fish_len
        self.school_z_sd = np.nanstd(school_zs, axis=0) / fish_len

    def remove_and_smooth_points(self):
        threshold = 0.01

        #Removing the points leaves too many gaps to then fill in later
        # median_school_center_x = medfilt(self.school_center_x, kernel_size = 11)
        # median_school_center_y = medfilt(self.school_center_y, kernel_size = 11)
        # median_school_center_z = medfilt(self.school_center_z, kernel_size = 11)

        # difference_x = np.abs(self.school_center_x - median_school_center_x)
        # difference_y = np.abs(self.school_center_y - median_school_center_y)
        # difference_z = np.abs(self.school_center_z - median_school_center_z)

        # outlier_id_x = difference_x > threshold
        # outlier_id_y = difference_y > threshold
        # outlier_id_z = difference_z > threshold

        # self.school_center_x[outlier_id_x] = np.nan
        # self.school_center_y[outlier_id_y] = np.nan
        # self.school_center_z[outlier_id_z] = np.nan

        self.school_center_x = savgol_filter(self.school_center_x,31,1)
        self.school_center_y = savgol_filter(self.school_center_y,31,1)
        self.school_center_z = savgol_filter(self.school_center_z,31,1)

    def calc_school_speed(self):
        #Based on the movement of the center of the school, not the mean of all the fish speeds

        #First we get the next points for the group
        group_x_next = np.roll(self.school_center_x, -1)
        group_y_next = np.roll(self.school_center_y, -1)
        group_z_next = np.roll(self.school_center_z, -1)

        #Then we create a vector of the future point minus the last one
        vec_x = group_x_next - self.school_center_x
        vec_y = group_y_next - self.school_center_y
        vec_z = group_z_next - self.school_center_z

        #Then we add the flow to the x value
        #Since (0,0) is in the upper left a positive vec_x value value means it is moving downstream
        #so I should subtract the flow value 
        #The flow value is mutliplied by the fish length since the vec_x values are in pixels, but it is in BLS so divide by fps
        vec_x_flow = vec_x - (self.flow*fish_len)/fps

        #It is divided in order to get it in body lengths and then times fps to get BL/s
        self.group_speed = np.sqrt(vec_x_flow**2+vec_y**2+vec_z**2)[:-1]/fish_len * fps

    def calc_school_tb_freq(self):
        tb_collect = []

        for fish in self.fishes:
            if len(fish.tb_freq_reps) > 0:
                tb_collect.append(fish.tb_freq_reps)

        self.group_tb_freq = np.nanmean(tb_collect, axis=0) 

    def calc_school_polarization(self):
        #formula from McKee 2020
        sin_headings = np.sin(np.deg2rad([fish.yaw_heading for fish in self.fishes]))
        cos_headings = np.cos(np.deg2rad([fish.yaw_heading for fish in self.fishes]))

        self.polarization = (1/self.n_fish)*np.sqrt(np.nansum(sin_headings, axis=0)**2 + np.nansum(cos_headings, axis=0)**2)

    def calc_nnd(self):
        #first we make an array to fill with the NNDs 
        nnd_array  = np.zeros((len(self.school_center_x),self.n_fish,self.n_fish)) + np.nan

        #now calculate all nnds
        for i in range(self.n_fish):
            for j in range(self.n_fish):

                if i != j:
                    fish1 = self.fishes[i]
                    fish2 = self.fishes[j]

                    dists = get_dist_np_2D(fish1.head_x,fish1.head_y,fish2.head_x,fish2.head_y)

                    #dists = get_dist_np_3D(fish1.head_x,fish1.head_y,fish1.head_z,fish2.head_x,fish2.head_y,fish2.head_z)

                    for t in range(len(self.school_center_x)):
                        nnd_array[t][i][j] = dists[t]

        #Then we get the mins of each row (or column, they are the same), and then get the mean for the mean
        # NND for that timepoint
        self.nearest_neighbor_distance = np.nanmean(np.nanmin(nnd_array,axis = 1),axis = 1) / fish_len

    def calc_tailbeat_cor(self):
        pass

    def calc_school_area(self):
        school_xs = np.asarray([fish.head_x for fish in self.fishes])
        school_ys = np.asarray([fish.head_y for fish in self.fishes])
        school_zs = np.asarray([fish.head_z for fish in self.fishes])

        self.school_areas = [np.nan for i in range(len(school_xs[0]))]

        for i in range(len(school_xs[0])):
            x_row = school_xs[:,i]
            y_row = school_ys[:,i]
            z_row = school_zs[:,i]

            mask = ~np.isnan(x_row)

            x_row = x_row[mask]
            y_row = y_row[mask]
            z_row = z_row[mask]

            mask = ~np.isnan(y_row)

            x_row = x_row[mask]
            y_row = y_row[mask]
            z_row = z_row[mask]

            mask = ~np.isnan(z_row)

            x_row = x_row[mask]
            y_row = y_row[mask]
            z_row = z_row[mask]

            if len(x_row) >= 4:
                points = np.column_stack((x_row,y_row,z_row))

                hull = ConvexHull(points)

                self.school_areas[i] = hull.volume/fish_len**2

    def calc_school_groups(self):
        min_BL_for_groups = 2

        school_xs = np.asarray([fish.head_x for fish in self.fishes])
        school_ys = np.asarray([fish.head_y for fish in self.fishes])
        school_zs = np.asarray([fish.head_z for fish in self.fishes])

        self.school_groups = [np.nan for i in range(len(school_xs[0]))]

        for i in range(len(school_xs[0])):

            x_row = school_xs[:,i]
            y_row = school_ys[:,i]
            z_row = school_zs[:,i]

            mask = ~np.isnan(x_row)

            x_row = x_row[mask]
            y_row = y_row[mask]
            z_row = z_row[mask]

            mask = ~np.isnan(y_row)

            x_row = x_row[mask]
            y_row = y_row[mask]
            z_row = z_row[mask]

            mask = ~np.isnan(z_row)

            x_row = x_row[mask]
            y_row = y_row[mask]
            z_row = z_row[mask]

            points = np.asarray([item for item in zip(x_row, y_row, z_row)])

            dm = np.zeros((len(points),len(points)))

            points = points/fish_len

            dm = distance_matrix(points,points)

            dm_min = dm <= min_BL_for_groups

            G = nx.from_numpy_array(dm_min)

            n_groups = len([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])

            self.school_groups[i] = n_groups

    def graph_values(self):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(ncols = 5, nrows = 3) 

        ax0 = plt.subplot(gs[:,0])
        for fish in self.fishes:
            ax0.scatter(fish.head_x, fish.head_y, c = np.linspace(0,1,num = len(fish.head_x)), s = 2)

        ax1 = plt.subplot(gs[0,1])
        ax1.plot(range(len(self.school_center_x)), self.school_center_x)
        ax1.set_title("School X")

        ax2 = plt.subplot(gs[1,1])
        ax2.plot(range(len(self.school_center_y)), self.school_center_y)
        ax2.set_title("School Y")

        ax2 = plt.subplot(gs[2,1])
        ax2.plot(range(len(self.school_center_z)), self.school_center_z)
        ax2.set_title("School Z")

        ax3 = plt.subplot(gs[0,2])
        ax3.plot(range(len(self.group_speed)), self.group_speed)
        ax3.set_title("School Speed")

        ax4 = plt.subplot(gs[1,2])
        ax4.plot(range(len(self.group_tb_freq)), self.group_tb_freq)
        ax4.set_title("School TB Freq")

        ax5 = plt.subplot(gs[2,2])
        ax5.plot(range(len(self.polarization)), self.polarization)
        ax5.set_title("School Polarization")

        ax6 = plt.subplot(gs[0,3])
        ax6.plot(range(len(self.school_areas)), self.school_areas)
        ax6.set_title("School Area")

        ax7 = plt.subplot(gs[1,3])
        ax7.plot(range(len(self.nearest_neighbor_distance)), self.nearest_neighbor_distance)
        ax7.set_title("School NND")

        plt.show()

class trial:
    def __init__(self, file_name, data_folder, n_fish = 8):
        self.file = file_name

        self.year = self.file[0:4]
        self.month = self.file[5:7]
        self.day = self.file[8:10]
        self.trial = self.file[11:13]
        self.abalation = self.file[15:16]
        self.darkness = self.file[18:19]
        self.flow = self.file[21:22]

        self.n_fish = n_fish
        self.data = pd.read_csv(data_folder+file_name, index_col=0, header=header)
        self.scorer = self.data.keys()[0][0]

        self.fishes = [fish_data("individual"+str(i+1),self.data,self.scorer,int(self.flow)) for i in range(n_fish)]

        #This sets the indexes so I can avoid any issues with having Fish 1 always be compared first 
        # and so on and so forth
        self.fish_comp_indexes = [[i,j] for i in range(n_fish) for j in range(i+1,n_fish)]

        for pair in self.fish_comp_indexes:
            random.shuffle(pair)

        self.fish_comps = [[0 for j in range(self.n_fish)] for i in range(self.n_fish)]

        #Now we fill in based on the randomized pairs
        for pair in self.fish_comp_indexes:
            self.fish_comps[pair[0]][pair[1]] = fish_comp(self.fishes[pair[0]],self.fishes[pair[1]])

        self.school_comp = school_comps(self.fishes, n_fish = n_fish, flow = int(self.flow))

    def return_trial_vals(self):
        print(self.year,self.month,self.day,self.trial,self.abalation,self.darkness,self.flow)

    def return_tailbeat_lens(self):
        all_tailbeat_lens = []

        for fish in self.fishes:
            all_tailbeat_lens.extend(np.diff(fish.zero_crossings))

        return all_tailbeat_lens

    def return_fish_lens(self):
        all_fish_lens = []

        for fish in self.fishes:
            all_fish_lens.extend(get_fish_length(fish))

        return all_fish_lens

    def return_fish_vals(self):
        firstfish = True

        for fish in self.fishes:

            chunked_yaw_headings = angular_mean_tailbeat_chunk(fish.yaw_heading,tailbeat_len)
            chunked_pitch_headings = angular_mean_tailbeat_chunk(fish.pitch_heading,tailbeat_len)
            chunked_speeds = mean_tailbeat_chunk(fish.speed,tailbeat_len)
            chunked_tb_freqs = mean_tailbeat_chunk(fish.tb_freq_reps,tailbeat_len)
            chunked_body_lengths = mean_tailbeat_chunk(fish.body_lengths,tailbeat_len)
            chunked_x = mean_tailbeat_chunk(fish.head_x,tailbeat_len)
            chunked_y = mean_tailbeat_chunk(fish.head_y,tailbeat_len)
            chunked_z = mean_tailbeat_chunk(fish.head_z,tailbeat_len)

            short_data_length = min([len(chunked_yaw_headings),len(chunked_pitch_headings),
                                     len(chunked_speeds),len(chunked_tb_freqs),
                                     len(chunked_x),len(chunked_y)])

            d = {'Year': np.repeat(self.year,short_data_length),
                 'Month': np.repeat(self.month,short_data_length),
                 'Day': np.repeat(self.day,short_data_length),
                 'Trial': np.repeat(self.trial,short_data_length), 
                 'Ablation': np.repeat(self.abalation,short_data_length), 
                 'Darkness': np.repeat(self.darkness,short_data_length), 
                 'Flow': np.repeat(self.flow,short_data_length), 
                 'Fish': np.repeat(fish.name,short_data_length),
                 'Tailbeat_Num': range(short_data_length),
                 'X':chunked_x[:short_data_length],
                 'Y':chunked_y[:short_data_length],
                 'Z':chunked_z[:short_data_length],
                 'Yaw Heading': chunked_yaw_headings[:short_data_length], 
                 'Pitch Heading': chunked_pitch_headings[:short_data_length], 
                 'Speed': chunked_speeds[:short_data_length], 
                 'TB_Frequency': chunked_tb_freqs[:short_data_length],
                 'Fish_Length': chunked_body_lengths[:short_data_length]}

            if firstfish:
                out_data = pd.DataFrame(data=d)
                firstfish = False
            else:
                out_data = out_data.append(pd.DataFrame(data=d))

        return(out_data)

    def return_comp_vals(self):
        firstfish = True

        for pair in self.fish_comp_indexes:

            current_comp = self.fish_comps[pair[0]][pair[1]]

            chunked_x_diffs = mean_tailbeat_chunk(current_comp.x_diff,tailbeat_len)
            chunked_y_diffs = mean_tailbeat_chunk(current_comp.y_diff,tailbeat_len)
            chunked_z_diffs = mean_tailbeat_chunk(current_comp.z_diff,tailbeat_len)
            chunked_dists = get_dist_np_3D(0,0,0,chunked_x_diffs,chunked_y_diffs,chunked_z_diffs)
            chunked_angles = mean_tailbeat_chunk(current_comp.angle,tailbeat_len)
            chunked_yaw_heading_diffs = angular_mean_tailbeat_chunk(current_comp.yaw_heading_diff,tailbeat_len)
            chunked_pitch_heading_diffs = angular_mean_tailbeat_chunk(current_comp.pitch_heading_diff,tailbeat_len)
            chunked_f1_speed = mean_tailbeat_chunk(current_comp.f1.speed,tailbeat_len)
            chunked_f2_speed = mean_tailbeat_chunk(current_comp.f2.speed,tailbeat_len)
            chunked_speed_diffs = mean_tailbeat_chunk(current_comp.speed_diff,tailbeat_len)
            chunked_tailbeat_offsets = mean_tailbeat_chunk(current_comp.tailbeat_offset_reps,tailbeat_len)

            short_data_length = min([len(chunked_x_diffs),len(chunked_y_diffs),len(chunked_dists),
                                     len(chunked_angles),len(chunked_yaw_heading_diffs),len(chunked_pitch_heading_diffs),
                                     len(chunked_speed_diffs),
                                     len(chunked_tailbeat_offsets)])

            d = {'Year': np.repeat(self.year,short_data_length),
                 'Month': np.repeat(self.month,short_data_length),
                 'Day': np.repeat(self.day,short_data_length),
                 'Trial': np.repeat(self.trial,short_data_length), 
                 'Ablation': np.repeat(self.abalation,short_data_length), 
                 'Darkness': np.repeat(self.darkness,short_data_length), 
                 'Flow': np.repeat(self.flow,short_data_length), 
                 'Fish': np.repeat(current_comp.name,short_data_length),
                 'Tailbeat_Num': range(short_data_length),
                 'X_Distance': chunked_x_diffs[:short_data_length], 
                 'Y_Distance': chunked_y_diffs[:short_data_length],
                 'Z_Distance': chunked_z_diffs[:short_data_length], 
                 'Distance': chunked_dists[:short_data_length],
                 'Angle': chunked_angles[:short_data_length],
                 'Yaw Heading_Diff': chunked_yaw_heading_diffs[:short_data_length],
                 'Pitch Heading_Diff': chunked_pitch_heading_diffs[:short_data_length],
                 'Fish1_Speed': chunked_f1_speed[:short_data_length],
                 'Fish2_Speed': chunked_f2_speed[:short_data_length],
                 'Speed_Diff': chunked_speed_diffs[:short_data_length],
                 'Sync': chunked_tailbeat_offsets[:short_data_length]}

            if firstfish:
                out_data = pd.DataFrame(data=d)
                firstfish = False
            else:
                out_data = out_data.append(pd.DataFrame(data=d))

        return(out_data)

    def return_raw_comp_vals(self):
        firstfish = True

        for pair in self.fish_comp_indexes:

            current_comp = self.fish_comps[pair[0]][pair[1]]

            dists = get_dist_np_3D(0,0,0,current_comp.x_diff,current_comp.y_diff,current_comp.z_diff)

            short_data_length = min([len(current_comp.x_diff),len(current_comp.y_diff),len(current_comp.z_diff),len(dists),
                                     len(current_comp.angle),len(current_comp.yaw_heading_diff),len(current_comp.pitch_heading_diff),
                                     len(current_comp.speed_diff),len(current_comp.tailbeat_offset_reps)])

            # print([len(current_comp.x_diff),len(current_comp.y_diff),len(dists),
            #                          len(current_comp.angle),len(current_comp.heading_diff),len(current_comp.speed_diff),
            #                          len(current_comp.tailbeat_offset_reps)])

            # print(short_data_length)
            # print(range(short_data_length))

            if short_data_length > tailbeat_len:

                d = {'Year': np.repeat(self.year,short_data_length),
                     'Month': np.repeat(self.month,short_data_length),
                     'Day': np.repeat(self.day,short_data_length),
                     'Trial': np.repeat(self.trial,short_data_length), 
                     'Ablation': np.repeat(self.abalation,short_data_length), 
                     'Darkness': np.repeat(self.darkness,short_data_length), 
                     'Flow': np.repeat(self.flow,short_data_length), 
                     'Fish': np.repeat(current_comp.name,short_data_length),
                     'Frame_Num': range(short_data_length),
                     'X_Distance': current_comp.x_diff[:short_data_length], 
                     'Y_Distance': current_comp.y_diff[:short_data_length], 
                     'Z_Distance': current_comp.z_diff[:short_data_length], 
                     'Distance': dists[:short_data_length],
                     'Angle': current_comp.angle[:short_data_length],
                     #Smoothing to make sure fish head wiggle doen't mess up polarization
                     'Fish1_Yaw_Heading': current_comp.f1.yaw_heading[:short_data_length],
                     'Fish1_Pitch_Heading': current_comp.f1.pitch_heading[:short_data_length],
                     'Fish2_Yaw_Heading': current_comp.f2.yaw_heading[:short_data_length],
                     'Fish2_Pitch_Heading': current_comp.f2.pitch_heading[:short_data_length],
                     'Yaw_Heading_Diff': current_comp.yaw_heading_diff[:short_data_length],
                     'Pitch_Heading_Diff': current_comp.pitch_heading_diff[:short_data_length],
                     'Fish1_Speed': current_comp.f1.speed[:short_data_length],
                     'Fish2_Speed': current_comp.f2.speed[:short_data_length],
                     'Speed_Diff': current_comp.speed_diff[:short_data_length],
                     'Sync': current_comp.tailbeat_offset_reps[:short_data_length]}

                if firstfish:
                    out_data = pd.DataFrame(data=d)
                    firstfish = False
                else:
                    out_data = out_data.append(pd.DataFrame(data=d))

        return(out_data)

    def return_school_vals(self):

        chunked_x_center = mean_tailbeat_chunk(self.school_comp.school_center_x,tailbeat_len)
        chunked_y_center = mean_tailbeat_chunk(self.school_comp.school_center_y,tailbeat_len)
        chunked_z_center = mean_tailbeat_chunk(self.school_comp.school_center_z,tailbeat_len)
        chunked_x_sd = mean_tailbeat_chunk(self.school_comp.school_x_sd,tailbeat_len)
        chunked_y_sd = mean_tailbeat_chunk(self.school_comp.school_y_sd,tailbeat_len)
        chunked_z_sd = mean_tailbeat_chunk(self.school_comp.school_z_sd,tailbeat_len)
        chunked_group_speed = mean_tailbeat_chunk(self.school_comp.group_speed,tailbeat_len)
        chunked_group_tb_freq = mean_tailbeat_chunk(self.school_comp.group_tb_freq,tailbeat_len)
        chunked_polarization = mean_tailbeat_chunk(self.school_comp.polarization,tailbeat_len)
        chunked_nnd = mean_tailbeat_chunk(self.school_comp.nearest_neighbor_distance,tailbeat_len)
        chunked_area = mean_tailbeat_chunk(self.school_comp.school_areas,tailbeat_len)
        chunked_groups = median_tailbeat_chunk(self.school_comp.school_groups,tailbeat_len)

        short_data_length = min([len(chunked_x_center),len(chunked_y_center),len(chunked_x_sd),
                                 len(chunked_y_sd),len(chunked_group_speed),len(chunked_group_tb_freq),
                                 len(chunked_nnd),len(chunked_area)])

        d = {'Year': np.repeat(self.year,short_data_length),
             'Month': np.repeat(self.month,short_data_length),
             'Day': np.repeat(self.day,short_data_length),
             'Trial': np.repeat(self.trial,short_data_length),
             'Tailbeat_Num': range(short_data_length),
             'Ablation': np.repeat(self.abalation,short_data_length), 
             'Darkness': np.repeat(self.darkness,short_data_length), 
             'Flow': np.repeat(self.flow,short_data_length), 
             'X_Center': chunked_x_center[:short_data_length], 
             'Y_Center': chunked_y_center[:short_data_length], 
             'Y_Center': chunked_z_center[:short_data_length],
             'X_SD': chunked_x_sd[:short_data_length], 
             'Y_SD': chunked_y_sd[:short_data_length], 
             'Z_SD': chunked_z_sd[:short_data_length], 
             'School_Polar': chunked_polarization[:short_data_length], 
             'School_Speed': chunked_group_speed[:short_data_length], 
             'School_TB_Freq': chunked_group_tb_freq[:short_data_length], 
             'NND': chunked_nnd[:short_data_length],
             'Area': chunked_area[:short_data_length],
             'Groups': chunked_groups[:short_data_length]}

        out_data = pd.DataFrame(data=d)

        return(out_data)

data_folder = "3D_Finished_Fish_Data_4P_gaps/"

trials = []

single_file = ""#"2021_10_06_36_LY_DN_F2_3D_DLC_dlcrnetms5_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv"

for file_name in os.listdir(data_folder):
    if file_name.endswith(".csv") and single_file in file_name:
        print(file_name)

        trials.append(trial(file_name,data_folder))

first_trial = True

#pair = trials[0].fish_comp_indexes[3]
#trials[0].fish_comps[pair[0]][pair[1]].graph_values()

print("Creating CSVs...")

for trial in trials:
    if first_trial:
        fish_sigular_dataframe = trial.return_fish_vals()
        fish_comp_dataframe = trial.return_comp_vals()
        fish_raw_comp_dataframe = trial.return_raw_comp_vals()
        fish_school_dataframe = trial.return_school_vals()
        first_trial = False
    else:
        fish_sigular_dataframe = fish_sigular_dataframe.append(trial.return_fish_vals())
        fish_comp_dataframe = fish_comp_dataframe.append(trial.return_comp_vals())
        fish_raw_comp_dataframe = fish_raw_comp_dataframe.append(trial.return_raw_comp_vals())
        fish_school_dataframe = fish_school_dataframe.append(trial.return_school_vals())

fish_sigular_dataframe.to_csv("Fish_Individual_Values_3D.csv")
fish_comp_dataframe.to_csv("Fish_Comp_Values_3D.csv")
fish_raw_comp_dataframe.to_csv("Fish_Raw_Comp_Values_3D.csv")
fish_school_dataframe.to_csv("Fish_School_Values_3D.csv")

# #Recalculate when new data is added
# all_trials_tailbeat_lens = []
# all_trials_fish_lens = []

# for trial in trials:
#     all_trials_tailbeat_lens.extend(np.asarray(trial.return_tailbeat_lens()))
#     all_trials_fish_lens.extend(np.asarray(trial.return_fish_lens()))

# all_trials_fish_lens = np.asarray(all_trials_fish_lens)
# #all_trials_fish_lens = all_trials_fish_lens[all_trials_fish_lens < 1.25]

# print("Tailbeat Len Median")
# print(np.nanmedian(all_trials_tailbeat_lens)) #18

# print("Fish Len Median")
# print(np.nanmedian(all_trials_fish_lens))
# print(np.exp(np.nanmedian(np.log(all_trials_fish_lens))))

# print("Fish Len Mean")
# print(np.nanmean(all_trials_fish_lens))
# print(np.exp(np.nanmean(np.log(all_trials_fish_lens))))

# print("Fish Len SD")
# print(np.nanstd(all_trials_fish_lens))
# print(np.nanstd(np.log(all_trials_fish_lens)))
# print(np.exp(np.nanstd(np.log(all_trials_fish_lens))))

# print("Fish Len Max?")
# print(np.nanmean(all_trials_fish_lens) + 3*np.nanstd(all_trials_fish_lens))
# print(np.exp(np.nanmean(np.log(all_trials_fish_lens))) + 3*np.exp(np.nanstd(np.log(all_trials_fish_lens))))

# print("Fish Len Max Observed")
# print(np.nanmax(all_trials_fish_lens))

# fig,ax = plt.subplots(1,2)
# ax[0].hist(all_trials_fish_lens, bins = 30)
# ax[1].hist(np.log(all_trials_fish_lens), bins = 30)
# #ax.set_xlim(0,1.5)
# plt.show()


# ish Len Median
# 0.06509714202072808
# Fish Len Mean
# 0.06978262158284285
# Fish Len SD
# 0.055777909686573444
# Fish Len Max?
# 0.23711635064256317

