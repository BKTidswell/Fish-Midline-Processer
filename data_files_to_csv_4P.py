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

from scipy.signal import hilbert, savgol_filter
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec 
import pandas as pd
import numpy as np
import math
import os

#Matplotlib breaks with Qt now in big sur :(
mpl.use('tkagg')

fps = 60

#The moving average window is more of a guess tbh
moving_average_n = 35

#Tailbeat len is the median of all frame distances between tailbeats
tailbeat_len = 19

#Fish len is the median of all fish lengths in pixels
fish_len = 193

#Header list for reading the raw location CSVs
header = list(range(4))

def get_dist_np(x1s,y1s,x2s,y2s):
    dist = np.sqrt((x1s-x2s)**2+(y1s-y2s)**2)
    return dist

def get_fish_length(fish):
    return get_dist_np(fish.head_x,fish.head_y,fish.midline_x,fish.midline_y) + get_dist_np(fish.midline_x,fish.midline_y,fish.tailbase_x,fish.tailbase_y) + get_dist_np(fish.tailbase_x,fish.tailbase_y,fish.tailtip_x,fish.tailtip_y)

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

class fish_data:
    def __init__(self, name, data, scorer):
        #This sets up all of the datapoints that I will need from this fish
        self.name = name
        self.head_x = data[scorer][name]["head"]["x"].to_numpy() 
        self.head_y = data[scorer][name]["head"]["y"].to_numpy() 

        self.midline_x = data[scorer][name]["midline2"]["x"].to_numpy() 
        self.midline_y = data[scorer][name]["midline2"]["y"].to_numpy() 

        self.tailbase_x = data[scorer][name]["tailbase"]["x"].to_numpy() 
        self.tailbase_y = data[scorer][name]["tailbase"]["y"].to_numpy() 

        self.tailtip_x = data[scorer][name]["tailtip"]["x"].to_numpy() 
        self.tailtip_y = data[scorer][name]["tailtip"]["y"].to_numpy()
        self.tailtip_perp = [] 
        
        #These are all blank and will be used for graphing
        self.normalized_tailtip = []
        self.tailtip_moving_average = []
        self.tailtip_zero_centered = []

        #These are the summary stats for the fish 
        self.heading = []
        self.speed = []
        self.zero_crossings = []
        self.tb_freq_reps = []

        #This calcualtes the summary stats
        self.calc_heading()
        self.calc_speed()
        self.calc_tailtip_perp()
        self.calc_tb_freq()

    #This function clacualtes the heading of the fish at each timepoint
    def calc_heading(self):
        #First we get the next points on the fish
        head_x_next = np.roll(self.head_x, -1)
        head_y_next = np.roll(self.head_y, -1)

        #Then we create a vector of the future point minus the last one
        vec_x = head_x_next - self.head_x
        vec_y = head_y_next - self.head_y

        #Then we use arctan to calculate the heading based on the x and y point vectors
        self.heading = np.rad2deg(np.arctan2(vec_y,vec_x))

    def calc_speed(self):
        #First we get the next points on the fish
        head_x_next = np.roll(self.head_x, -1)
        head_y_next = np.roll(self.head_y, -1)

        #You exclude the last value becuase of how it gets rolled over
        #It is divided in order to get it in body lengths and then times fps to get BL/s
        self.speed = get_dist_np(self.head_x,self.head_y,head_x_next,head_y_next)[:-1]/fish_len * fps

    def calc_tailtip_perp(self):

        #First get the total number of frames
        total_frames = len(self.head_x)

        out_tailtip_perps = []

        #My old code does this frame by frame. THere may be a way to vectorize it, but I'm not sure about that yet
        for i in range(total_frames):
            #Create a vector from the head to the tailtip and from the head to the midline
            tailtip_vec = np.asarray([self.head_x[i]-self.tailtip_x[i],self.head_y[i]-self.tailtip_y[i],0])
            midline_vec = np.asarray([self.head_x[i]-self.midline_x[i],self.head_y[i]-self.midline_y[i],0])

            #Then we make the midline vector a unit vector
            vecDist = np.sqrt(midline_vec[0]**2 + midline_vec[1]**2)
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
        tailbeat_lengths = abs(np.diff(np.append(self.zero_crossings,len(self.tailtip_zero_centered))))
        self.tb_freq_reps = np.repeat(tb_freq,tailbeat_lengths[:len(tb_freq)])

    #Thsi function allows me to graph values for any fish without trying to cram it into a for loop somewhere
    def graph_values(self):
        fig = plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(ncols = 3, nrows = 2) 

        ax0 = plt.subplot(gs[:,0])
        ax0.plot(self.head_x, self.head_y)
        ax0.plot(self.tailtip_x, self.tailtip_y)
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
        self.dist = []
        self.angle = []
        self.heading_diff = []
        self.speed_diff = []
        self.tailbeat_offset_reps = []

        self.calc_dist()
        self.calc_angle()
        self.calc_heading_diff()
        self.calc_speed_diff()
        self.calc_tailbeat_hilbert()

        #self.graph_values()

    def calc_dist(self):        
        #Divided to get it into bodylengths
        self.x_diff = (self.f1.head_x - self.f2.head_x)/fish_len
         #the y_diff is negated so it faces correctly upstream
        self.y_diff = -1*(self.f1.head_y - self.f2.head_y)/fish_len

        self.dist = get_dist_np(0,0,self.x_diff,self.y_diff)

    def calc_angle(self):
        #Calculate the angle of the x and y difference in degrees
        anglee_diff = np.rad2deg(np.arctan2(self.y_diff,self.x_diff))
        #This makes it from 0 to 360
        angles_diff_360 = np.mod(abs(anglee_diff-360),360)
        #This rotates it so that 0 is at the top and 180 is below the fish for a sideways swimming fish model
        self.angle = np.mod(angles_diff_360+90,360)

    def calc_heading_diff(self):
        self.heading_diff = np.rad2deg(np.arctan2(np.sin(self.f1.heading-self.f2.heading),
                                                  np.cos(self.f1.heading-self.f2.heading)))

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

        f1_tailtip_na_rm = self.f1.tailtip_zero_centered[~np.isnan(self.f1.tailtip_zero_centered)]
        f2_tailtip_na_rm = self.f2.tailtip_zero_centered[~np.isnan(self.f2.tailtip_zero_centered)]

        f1_analytic_signal = hilbert(f1_tailtip_na_rm)
        f1_instantaneous_phase = np.unwrap(np.angle(f1_analytic_signal))

        f2_analytic_signal = hilbert(f2_tailtip_na_rm)
        f2_instantaneous_phase = np.unwrap(np.angle(f2_analytic_signal))

        f1_instantaneous_phase_nan = np.zeros(self.f1.tailtip_zero_centered.shape)
        f1_instantaneous_phase_nan[f1_instantaneous_phase_nan == 0] = np.nan
        f1_instantaneous_phase_nan[~np.isnan(self.f1.tailtip_zero_centered)] = f1_instantaneous_phase

        f2_instantaneous_phase_nan = np.zeros(self.f2.tailtip_zero_centered.shape)
        f2_instantaneous_phase_nan[f2_instantaneous_phase_nan == 0] = np.nan
        f2_instantaneous_phase_nan[~np.isnan(self.f2.tailtip_zero_centered)] = f2_instantaneous_phase

        abs_diff_smooth = savgol_filter(abs(f2_instantaneous_phase_nan - f1_instantaneous_phase_nan),11,1)

        sync_slope = np.gradient(abs_diff_smooth)*2

        #norm_sync = np.power(2,abs(sync_slope)*-1)

        self.tailbeat_offset_reps = sync_slope


    def graph_values(self):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(ncols = 5, nrows = 3) 

        ax0 = plt.subplot(gs[:,0])
        ax0.plot(self.f1.head_x, self.f1.head_y)
        ax0.plot(self.f2.head_x, self.f2.head_y)
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
        ax6.plot(range(len(self.f1.heading)), self.f1.heading)
        ax6.set_title("Fish 1 Heading")

        ax7 = plt.subplot(gs[1,3])
        ax7.plot(range(len(self.f2.heading)), self.f2.heading)
        ax7.set_title("Fish 2 Heading")

        ax8 = plt.subplot(gs[2,3])
        ax8.plot(range(len(self.heading_diff)), self.heading_diff)
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

        self.fishes = [fish_data("individual"+str(i+1),self.data,self.scorer) for i in range(n_fish)]
        self.fish_comps = [[0 for j in range(self.n_fish)] for i in range(self.n_fish)]

        for i in range(self.n_fish):
            for j in range(i+1,self.n_fish):
                self.fish_comps[i][j] = fish_comp(self.fishes[i],self.fishes[j])

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

            chunked_headings = angular_mean_tailbeat_chunk(fish.heading,tailbeat_len)
            chunked_speeds = mean_tailbeat_chunk(fish.speed,tailbeat_len)
            chunked_tb_freqs = mean_tailbeat_chunk(fish.tb_freq_reps,tailbeat_len)

            short_data_length = min([len(chunked_headings),len(chunked_speeds),len(chunked_tb_freqs)])

            d = {'Year': np.repeat(self.year,short_data_length),
                 'Month': np.repeat(self.month,short_data_length),
                 'Day': np.repeat(self.day,short_data_length),
                 'Trial': np.repeat(self.trial,short_data_length), 
                 'Abalation': np.repeat(self.abalation,short_data_length), 
                 'Darkness': np.repeat(self.darkness,short_data_length), 
                 'Flow': np.repeat(self.flow,short_data_length), 
                 'Fish': np.repeat(fish.name,short_data_length),
                 'Tailbeat Num': range(short_data_length),
                 'Heading': chunked_headings[:short_data_length], 
                 'Speed': chunked_speeds[:short_data_length], 
                 'TB_Frequency': chunked_tb_freqs[:short_data_length]}

            if firstfish:
                out_data = pd.DataFrame(data=d)
                firstfish = False
            else:
                out_data = out_data.append(pd.DataFrame(data=d))

        return(out_data)

    def return_comp_vals(self):
        firstfish = True

        for i in range(self.n_fish):
            for j in range(i+1,self.n_fish):

                current_comp = self.fish_comps[i][j]

                chunked_x_diffs = mean_tailbeat_chunk(current_comp.x_diff,tailbeat_len)
                chunked_y_diifs = mean_tailbeat_chunk(current_comp.y_diff,tailbeat_len)
                chunked_dists = mean_tailbeat_chunk(current_comp.dist,tailbeat_len)
                chunked_angles = mean_tailbeat_chunk(current_comp.angle,tailbeat_len)
                chunked_heading_diffs = angular_mean_tailbeat_chunk(current_comp.heading_diff,tailbeat_len)
                chunked_speed_diffs = mean_tailbeat_chunk(current_comp.speed_diff,tailbeat_len)
                chunked_tailbeat_offsets = mean_tailbeat_chunk(current_comp.tailbeat_offset_reps,tailbeat_len)

                short_data_length = min([len(chunked_x_diffs),len(chunked_y_diifs),len(chunked_dists),
                                         len(chunked_angles),len(chunked_heading_diffs),len(chunked_speed_diffs),
                                         len(chunked_tailbeat_offsets)])

                d = {'Year': np.repeat(self.year,short_data_length),
                     'Month': np.repeat(self.month,short_data_length),
                     'Day': np.repeat(self.day,short_data_length),
                     'Trial': np.repeat(self.trial,short_data_length), 
                     'Abalation': np.repeat(self.abalation,short_data_length), 
                     'Darkness': np.repeat(self.darkness,short_data_length), 
                     'Flow': np.repeat(self.flow,short_data_length), 
                     'Fish': np.repeat(current_comp.name,short_data_length),
                     'Tailbeat Num': range(short_data_length),
                     'X_Distance': chunked_x_diffs[:short_data_length], 
                     'Y_Distance': chunked_y_diifs[:short_data_length], 
                     'Distance': chunked_dists[:short_data_length],
                     'Angle': chunked_angles[:short_data_length],
                     'Heading_Diff': chunked_heading_diffs[:short_data_length],
                     'Speed_Diff': chunked_speed_diffs[:short_data_length],
                     'Synchonization': chunked_tailbeat_offsets[:short_data_length]}

                if firstfish:
                    out_data = pd.DataFrame(data=d)
                    firstfish = False
                else:
                    out_data = out_data.append(pd.DataFrame(data=d))

        return(out_data)

data_folder = "Finished_Fish_Data_4P_gaps/"

trials = []

for file_name in os.listdir(data_folder):
    if file_name.endswith(".csv"):
        print(file_name)

        trials.append(trial(file_name,data_folder))

first_trial = True

print("Creating CSVs...")

for trial in trials:
    if first_trial:
        fish_sigular_dataframe = trial.return_fish_vals()
        fish_comp_dataframe = trial.return_comp_vals()
        first_trial = False
    else:
        fish_sigular_dataframe = fish_sigular_dataframe.append(trial.return_fish_vals())
        fish_comp_dataframe = fish_comp_dataframe.append(trial.return_comp_vals())

fish_sigular_dataframe.to_csv("Fish_Individual_Values.csv")
fish_comp_dataframe.to_csv("Fish_Comp_Values.csv")




#Recalculate when new data is added
# all_trials_tailbeat_lens = []
# all_trials_fish_lens = []

# for trial in trials:
#     all_trials_tailbeat_lens.extend(trial.return_tailbeat_lens())
#     all_trials_fish_lens.extend(trial.return_fish_lens())

# print("Tailbeat Len Median")
# print(np.nanmedian(all_trials_tailbeat_lens)) #19

# print("Fish Len Median")
# print(np.nanmedian(all_trials_fish_lens)) #193


