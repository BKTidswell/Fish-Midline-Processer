import matplotlib.pyplot as plt
from matplotlib import gridspec 
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pandas
import os
import numpy as np

header = list(range(4))

fish_names = ["individual1","individual2",
              "individual3","individual4",
              "individual5","individual6",
              "individual7","individual8"]

csv_output = "{Year},{Month},{Day},{Trial},{Ablation},{Darkness},{Singles},{Flow},{Turn_Dir},{Fish_Left},{Fish_Right}\n"

def turn_frames(x_data,y_data):
    diff_needed = 15
    smooth_window_size = 41

    x_data_smooth = savgol_filter(x_data, smooth_window_size, 3)
    y_data_smooth = savgol_filter(y_data, smooth_window_size, 3)

    x_data_rolled = np.roll(x_data_smooth, 5)
    y_data_rolled = np.roll(y_data_smooth, 5)

    x_diff = x_data_smooth - x_data_rolled
    y_diff = y_data_smooth - y_data_rolled

    angle = np.rad2deg(np.arctan2(y_diff,x_diff))

    angle_diff = angle - np.roll(angle, 1)

    angle_diff = (angle_diff + 180) % 360 - 180

    angle_diff = savgol_filter(angle_diff, smooth_window_size, 3)

    angle_diff_windowed = abs(np.convolve(angle_diff,np.ones(10,dtype=int),'valid'))

    peaks, _  = find_peaks(angle_diff_windowed, prominence = diff_needed)

    #-1 = Right, 1 = Left
    turn_dirs = np.sign(angle_diff[peaks])

    return(peaks,turn_dirs)

def is_point_LR(mid_to_head,mid_to_other):
    cross_prod = np.cross(mid_to_head,mid_to_other)

    if cross_prod[2] <= 0:
        return 0

    elif cross_prod[2] > 0:
        return 1

def get_num_LR(frame,main_fish,fish_df,scorerer):
    #index 0 is Right, index 1 is Left
    lr_out = [0,0]

    # print(frame)
    # print(fish_df[scorerer][main_fish]["head"]["x"])

    try:
        turn_Hx = fish_df[scorerer][main_fish]["head"]["x"][frame]
        turn_Hy = fish_df[scorerer][main_fish]["head"]["y"][frame]
        turn_Mx = fish_df[scorerer][main_fish]["midline2"]["x"][frame]
        turn_My = fish_df[scorerer][main_fish]["midline2"]["y"][frame]
    except:
        return [0,0]

    # print([turn_Hx,turn_Hy])
    # print([turn_Mx,turn_My])

    for fish in fish_names:
        if fish != main_fish:

            other_Hx = fish_df[scorerer][fish]["head"]["x"][frame]
            other_Hy = fish_df[scorerer][fish]["head"]["y"][frame]

            if not np.isnan(turn_Hx+turn_Hy+turn_Mx+turn_My+other_Hx+other_Hy):

                #print([other_Hx,other_Hy])

                m2h = [turn_Hx - turn_Mx,turn_Hy - turn_My,0]
                m2o = [other_Hx - turn_Mx,other_Hy - turn_My,0]

                LR_p = is_point_LR(m2h,m2o)

                lr_out[LR_p] += 1

    return(lr_out)

def process_trial(folder,datafile):
    fish_data = pandas.read_csv(folder+datafile,index_col=0, header=header)

    year = datafile[0:4]
    month = datafile[5:7]
    day = datafile[8:10]
    trial = datafile[11:13]
    abalation = datafile[15:16]
    darkness = datafile[18:19]
    flow = datafile[21:22]

    scorerer = fish_data.keys()[0][0]

    for fish in fish_names:
        x_data = fish_data[scorerer][fish]["head"]["x"]
        y_data = fish_data[scorerer][fish]["head"]["y"]

        turning_frames, turn_dirs = turn_frames(x_data,y_data)

        for i,frame in enumerate(turning_frames):
            num_LR = get_num_LR(frame,"individual2",fish_data,scorerer)

            f.write(csv_output.format(Year = year,
                                      Month = month,
                                      Day = day,
                                      Trial = trial,
                                      Ablation = abalation,
                                      Darkness = darkness,
                                      Singles = "N",
                                      Flow = flow,
                                      Turn_Dir = turn_dirs[i],
                                      Fish_Left = num_LR[1],
                                      Fish_Right = num_LR[0]))

f = open("Single_Fish_Data.csv", "w")

f.write("Year,Month,Day,Trial,Ablation,Darkness,Singles,Flow,Turn_Dir,Fish_Left,Fish_Right")

folder = "Finished_Fish_Data_4P_gaps/"

for file_name in os.listdir(folder):
    if file_name.endswith(".csv"):
        print(file_name)
        
        process_trial(folder,file_name)

f.close()


