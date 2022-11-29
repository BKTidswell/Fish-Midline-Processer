
#The point of this code is to get the fish labled in the same order
# Since if fish 1 in V1 and V2 isn't the same fish, it will all be a real mess
#I will attempt to do this by getting x and y values from the head of the fish, and labeling them from 1 to 8
# up to down and left to right. This will require finding a frame where all those fish are present, as I don't think
# all these videos have a full first frames. But let's find out!

import pandas as pd
import numpy as np
import os

#Header list for reading the raw location CSVs
header = list(range(4))

#Get all te files
v1_files = os.listdir("V1 CSVs")
v2_files = os.listdir("V2 CSVs")

num_fish = 8
body_parts = ["head","midline2","tailbase","tailtip"]


# We have more v1 files than v2, so we do this for every v2 file
for v2f in v2_files:
    if v2f.endswith(".csv"):

        #Get a long ID for the matching V1, short ID for the DLT
        file_id = v2f[0:22]
        short_id = v2f[0:10]

        print(file_id,short_id)

        #Get the v1 file that matches, and the dlt coefs that go with them both
        v1f = [f for f in v1_files if file_id in f][0]

        #Add the filepath on here as well
        v1f = "V1 CSVs/" + v1f
        v2f = "V2 CSVs/" + v2f

        print(v1f,v2f)

        #Read in the raw data
        v1_raw_data = pd.read_csv(v1f, index_col=0, header=header)
        v2_raw_data = pd.read_csv(v2f, index_col=0, header=header)

        v1_scorer = v1_raw_data.keys()[0][0]
        v2_scorer = v2_raw_data.keys()[0][0]

        data_length = len(v1_raw_data)

        print(data_length)

        both_all_fish_ind = -1

        for i in range(1,data_length):
            row_sum_v1 = np.sum(v1_raw_data.values[i-1:i])
            row_sum_v2 = np.sum(v2_raw_data.values[i-1:i])

            if not np.isnan(row_sum_v1+row_sum_v2):
                both_all_fish_ind = i

                break

        print(both_all_fish_ind)

        #Well... turns out that this might not work since there's not always a time when both have all fish for all data
        # So I'll talk to Eric and we'll see
        






