import os
import numpy as np
import pandas as pd
import math
import plotly.express as px
from scipy import linalg
import sys

#Header list for reading the raw location CSVs
header = list(range(4))

num_fish = 8
body_parts = ["head","midline2","tailbase","tailtip"]

file_paths = ["3D_Finished_Fish_Data_4P_gaps/","Finished_Fish_Data_4P_gaps/"]

files_3D = os.listdir("3D_Finished_Fish_Data_4P_gaps")
files_2D = os.listdir("Finished_Fish_Data_4P_gaps")


for path in file_paths:

    #Now finally get the percent of "good" points
    files = []
    good_percents = []

    for file_name in os.listdir(path):
        if file_name.endswith("filtered.csv"):

            in_csv = open(path+file_name,"r")

            #Skip the first 1
            skip_lines = 0

            good_points = 0
            total_points = 0

            for line in in_csv:
                if skip_lines >= 4:
                    lis = line.split(",")

                    #Off by 1 for the first row of frame counts
                    probs = np.asarray(lis[1::3])
                    probs = np.where(probs == "\n", 0, probs)
                    probs = np.where(probs == "", 0, probs).astype(np.float)

                    #If there are no zeros in probs
                    good_points += np.sum(probs != 0)
                    total_points += len(probs)

                else:
                    skip_lines += 1

                    lis = line.split(",")[:-24]

            files.append(file_name)
            good_percents.append(round(good_points/total_points*100,2))

            in_csv.close()
            #out_csv.close()

    files = np.asarray(files)
    good_percents = np.asarray(good_percents)

    sort_percent = np.flip(good_percents[np.argsort(good_percents)])
    sorted_files = np.flip(files[np.argsort(good_percents)])

    #out_file = open(path+"Percent_Good.txt","w")

    # for i in range(len(sorted_files)):
    #     out_file.write("{} : {}% good points\n".format(sorted_files[i][:33],sort_percent[i]))
    #     print("{} : {}% good points".format(sorted_files[i][:33],sort_percent[i]))

    print(path)
    print("Mean Good Points: {}".format(np.mean(good_percents)))

    #out_file.close()
