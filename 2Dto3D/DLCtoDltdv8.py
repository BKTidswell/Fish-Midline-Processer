import numpy as np
import pandas as pd
import math

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
    time_points = len(fish_data[cols[1]])

    for i in range(0,len(cols)-1):
        #Every 3 [x,y,p] times number of body parts columns we move onto a new fish
        fish_num = math.floor(i/(n_b_parts*3))
        #Every 3 [x,y,p] columns we move to a new body part, and we rotate through all
        fish_part = fish_parts[int(math.floor(i/3)%n_b_parts)]
        #rotate through the 3 [x,y,p] data type columns
        data_point = data_points[i%3]
        #Store the column data in the dict
        #Ignore the first 3 rows as those are column labels
        #print(fish_num,fish_part,data_point)

        fish_dict[fish_num][fish_part][data_point] = fish_data[cols[i+1]][3:time_points].astype(float).to_numpy()

    return(fish_dict,time_points-3)

n_fish = 8
b_parts_csv = ["head","tailbase","midline2","tailtip"]
n_b_parts = len(b_parts_csv)
b_parts = ["head","midline2","tailbase","tailtip"]

fish_dict_V1,time_points_V1 = DLC_CSV_to_dict(num_fish = n_fish, fish_parts = b_parts_csv, file = "2020_07_28_11_TN_DN_F0_V1DLC_resnet50_L8FVJul4shuffle1_100000_bx_filtered.csv")
fish_dict_V2,time_points_V2 = DLC_CSV_to_dict(num_fish = n_fish, fish_parts = b_parts_csv, file = "2020_07_28_11_TN_DN_F0_V2DLC_resnet50_L8FV4PFeb22shuffle1_100000_bx_filtered.csv")

#So it needs 4 files but most of them can basically be blank
#This one has xy points
dlt_xy_pts_dict = {}
#This can have zero for each camera
dlt_offsets_dict = {"cam1_offset":np.zeros(time_points_V1),"cam2_offset":np.zeros(time_points_V1)}
#This can be all NaN
dlt_xyz_pts_dict = {}
#This can be all NaN
dlt_xyz_res_dict = {}

cams = ["cam1","cam2"]

dlt_xy_pts_header = "pt{num}_{cam}_{p}"
dlt_xyz_pts_header = "pt{num}_{p}"
dlt_xyz_res_header = "pt{num}_dltres"

for i in range(n_fish):
    for j, part in enumerate(b_parts):
        dlt_xyz_pts_dict[dlt_xyz_pts_header.format(num = i*n_b_parts+j+1, p = "X")] = ["NaN"] * time_points_V1
        dlt_xyz_pts_dict[dlt_xyz_pts_header.format(num = i*n_b_parts+j+1, p = "Y")] = ["NaN"] * time_points_V1

        dlt_xyz_res_dict[dlt_xyz_res_header.format(num = i*n_b_parts+j+1)] = ["NaN"] * time_points_V1

        for k, data in enumerate([fish_dict_V1, fish_dict_V2]):
            for point in ["x","y"]:
                dlt_xy_pts_dict[dlt_xy_pts_header.format(num = i*n_b_parts+j+1, cam = cams[k], p = point.upper())] = data[i][part][point]

xy_pts_df = pd.DataFrame(dlt_xy_pts_dict)
xy_pts_df.to_csv ("2020_07_28_11_TN_DN_F0_xypts.csv", index = False, header=True)

offsets_df = pd.DataFrame(dlt_offsets_dict)
offsets_df.to_csv ("2020_07_28_11_TN_DN_F0_offsets.csv", index = False, header=True)

xyz_pts_df = pd.DataFrame(dlt_xyz_pts_dict)
xyz_pts_df.to_csv ("2020_07_28_11_TN_DN_F0_xyzpts.csv", index = False, header=True)

xyz_res_df = pd.DataFrame(dlt_xyz_res_dict)
xyz_res_df.to_csv ("2020_07_28_11_TN_DN_F0_xyzres.csv", index = False, header=True)