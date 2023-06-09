import matplotlib.pyplot as plt
from matplotlib import gridspec 
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pandas
import os
import numpy as np
import math

header = list(range(4))

fish_names = ["individual1","individual2",
              "individual3","individual4",
              "individual5","individual6",
              "individual7","individual8"]

x_edges = [250,2250]
y_edges = [200,900]

csv_output = "{Year},{Month},{Day},{Trial},{Ablation},{Darkness},{Singles},{Flow},{Frame},{Fish},{Turn_Dir},{Fish_Left},{Fish_Right}\n"

def calc_mag(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def turn_frames(head_x_data,head_y_data,mid_x_data,mid_y_data):

    #Get the head and midline data in a form that is easy to subtract
    head_point_data = np.column_stack((head_x_data, head_y_data))
    mid_point_data = np.column_stack((mid_x_data, mid_y_data))

    #Set up an array for dot products
    dot_prods = np.zeros(len(head_point_data))+1

    #This is just chosen randomly based on what I've seen
    offset = 20

    for i in range(len(head_point_data)-offset-1):

        vec1 = (head_point_data[i] - mid_point_data[i]) / calc_mag(head_point_data[i],mid_point_data[i])
        vec2 = (head_point_data[i+offset] - mid_point_data[i+offset]) / calc_mag(head_point_data[i+offset],mid_point_data[i+offset])

        dot_prods[i] = np.dot(vec1,vec2)

        #print(vec1, vec2, dot_prods[i])

        if np.isnan(np.dot(vec1,vec2)):
            dot_prods[i] = 1

    #Flip the dot products around so that higher more of a turn, not less
    dot_prods = abs(dot_prods-1)

    #Get the moving average. It's a window I just chose, but it works well I suppose
    dot_prods = moving_average(dot_prods,10)

    #Instead of setting an arbitray amount, look at ones that are more sificantly different from the rest. 
    #Though I suppose that if they turned a lot this wouldn't work well...
    #okay commenting it out for now and just picking a number

    #peak_prom = np.std(dot_prods)*1.5
    peak_prom = 0.3

    #Now zero out all the areas less than the peak prom
    dot_prods_over_min = np.where(dot_prods<=peak_prom,0,1)*dot_prods

    #And then find the maxes in those non zeroed areas
    peaks, _  = find_peaks(dot_prods_over_min, prominence = peak_prom)

    # 0 is Right, index 1 is Left
    #Oh but now I need to find a way to see if they turned left or right
    # and dot product doesn't really do that.

    #But I can do it with is_point_LR() for each point, comparing one head to the next...
    turn_dirs = []

    final_peaks = []

    for p in peaks:
        #mid to head and mid to next
        m2h = np.pad(head_point_data[p] - mid_point_data[p], (0, 1), 'constant')
        m2n = np.pad(head_point_data[p+offset] - mid_point_data[p], (0, 1), 'constant')

        turn_dir = is_point_LR(m2h,m2n)

        if turn_dir == None:
            print(m2h,m2n)

        #Also we want to check that they aren't too close to the edge!
        if((head_point_data[p][0] > x_edges[0]) and (head_point_data[p][0] < x_edges[1]) and (head_point_data[p][1] > y_edges[0]) and (head_point_data[p][1] < y_edges[1])):
            turn_dirs.append(turn_dir)
            final_peaks.append(p)
        else:
            print("Edge case!")

    return(final_peaks,turn_dirs)

def is_point_LR(mid_to_head,mid_to_other):
    #Get the cross product to see if they turned left or right
    #0 is Right, index 1 is Left
    cross_prod = np.cross(mid_to_head,mid_to_other)

    #print(cross_prod)

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
        head_x_data = fish_data[scorerer][fish]["head"]["x"].to_numpy()
        head_y_data = fish_data[scorerer][fish]["head"]["y"].to_numpy()

        mid_x_data = fish_data[scorerer][fish]["midline2"]["x"].to_numpy()
        mid_y_data = fish_data[scorerer][fish]["midline2"]["y"].to_numpy()

        turning_frames, turn_dirs = turn_frames(head_x_data,head_y_data,mid_x_data,mid_y_data)

        for i,frame in enumerate(turning_frames):
            num_LR = get_num_LR(frame,fish,fish_data,scorerer)

            f.write(csv_output.format(Year = year,
                                      Month = month,
                                      Day = day,
                                      Trial = trial,
                                      Ablation = abalation,
                                      Darkness = darkness,
                                      Singles = "N",
                                      Flow = flow,
                                      Frame = frame,
                                      Fish = fish,
                                      Turn_Dir = turn_dirs[i],
                                      Fish_Left = num_LR[1],
                                      Fish_Right = num_LR[0]))

f = open("single_fish_turning.csv", "w")

f.write("Year,Month,Day,Trial,Ablation,Darkness,Singles,Flow,Frame,Fish,Turn_Dir,Fish_Left,Fish_Right\n")

folder = "Single_Fish_Data/"

for file_name in os.listdir(folder):
    if file_name.endswith(".csv"):
        print(file_name)
        
        process_trial(folder,file_name)

f.close()


