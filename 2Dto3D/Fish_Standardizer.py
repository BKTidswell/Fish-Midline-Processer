
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

#Get all the files
v1_files = os.listdir("V1 CSVs")
v2_files = os.listdir("V2 CSVs")

new_data_folder = "V2 CSVs Adjusted/"

num_fish = 8
body_parts = ["head","midline2","tailbase","tailtip"]

def get_dist_np(x1s,y1s,x2s,y2s):
    dist = np.sqrt((x1s-x2s)**2+(y1s-y2s)**2)
    return dist

def create_index_col_old(indexes):
    index_col = []

    for ind in indexes:
        index_col.append("individual"+str(ind))

    index_col.append("single")

    return(np.asarray(index_col))

def create_index_col(v1_ind, v2_ind):
    index_col = ["" for i in range(num_fish+1)]

    for i in range(num_fish):
        index_col[v1_ind[i]-1] = "individual"+str(v2_ind[i])

    index_col[num_fish] = "single"

    return(index_col)


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

            #We only need there to always be head data, we don't need all the points, so this can still work
            #Also this is how best to slice a multiindex that's just multicolumns, this data is weird
            v1_raw_head_data = v1_raw_data.xs('head',axis=1,level=2)
            v2_raw_head_data = v2_raw_data.xs('head',axis=1,level=2)

            row_sum_v1 = np.sum(v1_raw_head_data.values[i-1:i])
            row_sum_v2 = np.sum(v2_raw_head_data.values[i-1:i])

            if not np.isnan(row_sum_v1+row_sum_v2):
                both_all_fish_ind = i

                break

        #Okay so now we have an index where all fish heads exist in both views, so we can then we can sort them 
        # Based on distance to origin, and since they are only in one quadrent this should work to seperate them
        print(both_all_fish_ind)

        #So now we get the head xs and ys for both
        v1_head_x = v1_raw_data.xs('head',axis=1,level=2).xs('x',axis=1,level=2).values[both_all_fish_ind-1:both_all_fish_ind][0]
        v1_head_y = v1_raw_data.xs('head',axis=1,level=2).xs('y',axis=1,level=2).values[both_all_fish_ind-1:both_all_fish_ind][0]

        v2_head_x = v2_raw_data.xs('head',axis=1,level=2).xs('x',axis=1,level=2).values[both_all_fish_ind-1:both_all_fish_ind][0]
        v2_head_y = v2_raw_data.xs('head',axis=1,level=2).xs('y',axis=1,level=2).values[both_all_fish_ind-1:both_all_fish_ind][0]

        #And then the distances from the origins
        v1_head_dists = get_dist_np(0,0,v1_head_x,v1_head_y)
        v2_head_dists = get_dist_np(0,0,v2_head_x,v2_head_y)

        #Then we make a dummy order so we can tell how the points match up
        v1_order = np.arange(num_fish)+1
        v2_order = np.arange(num_fish)+1

        #Then sort these in order to find  how they match up
        v1_head_dists_sorted = v1_head_dists.argsort()
        v2_head_dists_sorted = v2_head_dists.argsort()

        v1_order_sorted = v1_order[v1_head_dists_sorted]
        v2_order_sorted = v2_order[v2_head_dists_sorted]

        print(v1_order_sorted,v2_order_sorted)

        #Now make the new order by using the order of one as the indexes of the other
        # Basically swapping points so that they all match with each other based on the distances

        #Honestly this took me a bit to figure out. But say you swapped 2 and 6 between them.
        # You would make the 2nd index of V2 6, and the 6th index into 2

        # new_order[v1[i]] = v2[i]

        #Obviosuly some are more swaps than jsut 1 for 1, but doing this makes sure they match up correctly

        #And then return that as an array
        new_v2_order = create_index_col(v1_order_sorted,v2_order_sorted)

        #Set the new index levels
        v2_raw_data.columns = v2_raw_data.columns.set_levels(new_v2_order, level=1)

        #And then save
        new_file_name = v2f.replace("V2 CSVs/",new_data_folder)
        v2_raw_data.to_csv("/Users/Ben/Desktop/Fish-Midline-Processer/2Dto3D/"+new_file_name)



        






