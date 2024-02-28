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

# x_edges = [250,2250]
# y_edges = [200,900]

# x_edges = [-0.05,0.75]
# y_edges = [-0.05,0.27]

#For Real Fish
# wall_lines = np.array([[[-0.05,0.27],[0.75,0.27]],
#                        [[0.75,0.27],[0.75,-0.05]],
#                        [[0.75,-0.05],[-0.05,-0.05]],
#                        [[-0.05,-0.05],[-0.05,0.27]]])

# #For Single Fish
wall_lines = np.array([[[0,1000],[2250,1000]],
                       [[2250,1000],[2250,0]],
                       [[2250,0],[0,0]],
                       [[0,0],[0,1000]]])

#Tailbeat len is the median of all frame distances between tailbeats
tailbeat_len = 19


#calculate the dot product value needed to have the minimum turn
min_turn_angle = 30
min_turn_radian = min_turn_angle * np.pi / 180
peak_prom = abs(np.dot((1,0),(np.cos(min_turn_radian), np.sin(min_turn_radian)))-1)/2


csv_output = "{Year},{Month},{Day},{Trial},{Ablation},{Darkness},{Singles},{Flow},{Frame},{Fish},{Turn_Dir},{Fish_Left},{Fish_Right},{Wall_Dist}\n"

def calc_mag(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def calc_mag_multi(p1,p2):
    return np.asarray([math.sqrt((p1[i][0]-p2[i][0])**2 + (p1[i][1]-p2[i][1])**2) for i in range(len(p1))])

def calc_mag_vec(v1):
    return math.sqrt((v1[0])**2 + (v1[1])**2)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def moving_sum(x, w):
    return np.convolve(x, np.ones(w), 'same')

def closest_right_distance_to_wall(headX, headY, midlineX, midlineY):

    #find all the distances from every wall
    distances = np.full([4], np.nan)
    percents = np.full([4], np.nan)

    for i,wall in enumerate(wall_lines):

        #See math document from Eric 
        #If you're not me, email me I guess?
        #But this gives k and b 
        # where k is the mutliplier of vector n that makes it intersect with vector w (wall)
        # and where b is the mutliplier of vector w that makes it intersect with vector n (right perpendicular to fish)
        nX = (headY - midlineY)
        nY = -1*(headX - midlineX)

        a = np.array([[wall[0][0] - headX],
                      [wall[0][1] - headY]])

        b = np.array([[nX , wall[0][0] - wall[1][0]],
                      [nY , wall[0][1] - wall[1][1]]])

        invB = np.linalg.inv(b)
        answer = np.matmul(invB,a)

        #This gives us the distance on the right the fish is from the wall
        #And the percent along the wall that line intersects with
        distances[i] = answer[0][0] * np.sqrt(nX**2 + nY**2)
        percents[i] = answer[1][0]

    #Then we only get the none negative distances and percents (the real ones)
    real_distances = (distances * (distances >= 0) * (percents >= 0) * (percents <= 1))

    #Then find the closest distance o those, and the percent that goes along with it
    closest_real_dist = np.nanmin(np.where(real_distances == 0, np.nan, real_distances))

    #If the fish is out of bounds return nan
    if np.isnan(closest_real_dist):
        return(np.nan,np.nan)

    closest_real_percent = percents[np.where(distances == closest_real_dist)[0][0]]

    return(closest_real_dist, closest_real_percent)


def turn_frames(head_x_data,head_y_data,mid_x_data,mid_y_data):

    #Get the head and midline data in a form that is easy to subtract
    head_point_data = np.column_stack((head_x_data, head_y_data))
    mid_point_data = np.column_stack((mid_x_data, mid_y_data))

    #Set up an array for dot products
    dot_prods = np.zeros(len(head_point_data))+1

    for i in range(tailbeat_len,len(head_point_data)-tailbeat_len-1):

        vec_20_before = (head_point_data[i-tailbeat_len:i] - mid_point_data[i-tailbeat_len:i]) / calc_mag_multi(head_point_data[i-tailbeat_len:i],mid_point_data[i-tailbeat_len:i]).reshape(tailbeat_len,1)
        vec_20_after = (head_point_data[i:i+tailbeat_len] - mid_point_data[i:i+tailbeat_len]) / calc_mag_multi(head_point_data[i:i+tailbeat_len],mid_point_data[i:i+tailbeat_len]).reshape(tailbeat_len,1)

        vec_20_before_avg = np.average(vec_20_before, axis = 0)
        vec_20_after_avg = np.average(vec_20_after, axis = 0)

        vec_20_before_avg_unit = vec_20_before_avg / calc_mag_vec(vec_20_before_avg)
        vec_20_after_avg_unit = vec_20_after_avg / calc_mag_vec(vec_20_after_avg)

        dot_prods[i] = np.dot(vec_20_before_avg_unit,vec_20_after_avg_unit)

        if np.isnan(dot_prods[i]):
            dot_prods[i] = 1

    #Flip the dot products around so that higher more of a turn, not less
    dot_prods = abs(dot_prods-1)/2

    #Get the moving average. It's a window I just chose, but it works well I suppose
    #No more average since we're averaging the vectors over a tailbeat
    #dot_prods_avg = moving_average(dot_prods,10)

    #Gethe moving sum over one second
    #dot_prods_sum = moving_sum(dot_prods_avg,60)

    #print(len(dot_prods),len(dot_prods_avg),len(dot_prods_sum))

    #Instead of setting an arbitray amount, look at ones that are more sificantly different from the rest. 
    #Though I suppose that if they turned a lot this wouldn't work well...
    #Now it represents 45 degrees

    #peak_prom = np.std(dot_prods)*1.5
    #peak_prom = 0.146

    #Now zero out all the areas less than the peak prom
    dot_prods_over_min = np.where(dot_prods<=peak_prom,0,1)*dot_prods

    #And then find the maxes in those non zeroed areas
    peaks, _  = find_peaks(dot_prods_over_min, prominence = peak_prom)

    #Graphing
    # fig = plt.figure(figsize=(8, 6))

    # gs = gridspec.GridSpec(ncols = 2, nrows = 2) 

    # ax0 = plt.subplot(gs[:,0])
    # ax0.plot(head_x_data, head_y_data)
    # ax0.plot(mid_x_data, mid_y_data)
    # ax0.plot(head_x_data[peaks], head_y_data[peaks], "x")
    # ax1 = plt.subplot(gs[:,1])

    # ax1.plot(np.arange(len(dot_prods)), dot_prods)
    # #ax1.plot(np.arange(len(dot_prods_avg)), dot_prods_avg)
    # #ax1.plot(np.arange(len(dot_prods_sum)), dot_prods_sum)
    # #Works best to display
    # ax1.plot(peaks, dot_prods[peaks], "x")
    # ax1.hlines(y=peak_prom, xmin=0, xmax=len(dot_prods), linewidth=1, color='r')
    # #ax1.set(ylim=(0, 0.5))

    # plt.show()

    # 0 is Right, index 1 is Left
    #Oh but now I need to find a way to see if they turned left or right
    # and dot product doesn't really do that.

    #But I can do it with is_point_LR() for each point, comparing one head to the next...
    turn_dirs = []
    final_peaks = []
    wall_dists = []

    for p in peaks:
        #mid to head and mid to next
        m2h = np.pad(head_point_data[p] - mid_point_data[p], (0, 1), 'constant')
        m2n = np.pad(head_point_data[p+tailbeat_len] - mid_point_data[p], (0, 1), 'constant')

        turn_dir = is_point_LR(m2h,m2n)

        if turn_dir == None:
            print(m2h,m2n)
        #We only want to run this if we know what the direction of the turn is
        else:
            #Also we want to check that they aren't too close to the edge!
            #Now we don't check! We're jsut going to calculate it!
            distance, percent = closest_right_distance_to_wall(head_point_data[p][0],head_point_data[p][1],mid_point_data[p][0],mid_point_data[p][1])

            #Only pass it on if in bounds
            if not np.isnan(distance):
                turn_dirs.append(turn_dir)
                final_peaks.append(p)
                wall_dists.append(distance)
            else:
                print("Out of Bounds!!")
                print(head_point_data[p])
                print(mid_point_data[p])
                print("")


    return(final_peaks,turn_dirs,wall_dists)


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

        turning_frames, turn_dirs, wall_dists = turn_frames(head_x_data,head_y_data,mid_x_data,mid_y_data)

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
                                      Fish_Right = num_LR[0],
                                      Wall_Dist = wall_dists[i]))

f = open("single_fish_data_turning.csv", "w")

f.write("Year,Month,Day,Trial,Ablation,Darkness,Singles,Flow,Frame,Fish,Turn_Dir,Fish_Left,Fish_Right,Wall_Dist\n")

folder = "single_Fish_Data/"

for file_name in os.listdir(folder):
    if file_name.endswith(".csv"):
        print(file_name)
        
        process_trial(folder,file_name)

f.close()


