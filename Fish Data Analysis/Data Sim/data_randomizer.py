import pandas as pd

#Header list for reading the raw location CSVs
header = list(range(4))

#Fish 1 is columns 0 to 11
#Fish 2 is columns 12 to 23
#Fish 3 is columns 24 to 35
#Fish 4 is columns 36 to 47
#Fish 5 is columns 48 to 59
#Fish 6 is columns 60 to 71
#Fish 7 is columns 72 to 83
#Fish 8 is columns 84 to 95

data_folder = "Finished_Fish_Data_4P_gaps_randomized/"

files = ["2020_06_29_20_TN_DN_F0_VDLC_resnet50_L8FV4PFeb22shuffle1_100000_bx_filtered.csv",
"2020_07_28_10_TN_DN_F0_V1DLC_resnet50_L8FV4PFeb22shuffle1_100000_bx_filtered.csv",
"2020_07_28_11_TN_DN_F0_V1DLC_dlcrnetms5_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv",
"2020_07_28_24_TN_DN_F0_V1DLC_dlcrnetms5_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv",
"2021_03_25_01_TN_DN_F0_V1DLC_resnet50_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv",
"2021_03_25_10_TN_DN_F0_V1DLC_resnet50_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv",
"2021_03_25_12_TN_DN_F0_V1DLC_dlcrnetms5_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv",
"2021_03_25_22_TN_DN_F0_V1DLC_dlcrnetms5_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv",
"2021_03_25_24_TN_DN_F0_V1DLC_dlcrnetms5_DLC_2-2_4P_8F_Light_VentralMay10shuffle1_100000_el_filtered.csv"]

data_to_fill = pd.read_csv(data_folder+files[0], index_col=0, header=header)

all_data_arr = [pd.read_csv(data_folder+f, index_col=0, header=header) for f in files]

#We will make 8 piecemeal dataframes
for i in range(8):
    #For each we will pull from 8 dataframes
    for j in range(8):
        #print(i,j,j*12,(j+1)*12-1)
        data_to_fill.iloc[: , j*12:(j+1)*12-1] = all_data_arr[(j+i)%8].iloc[: , j*12:(j+1)*12-1]

    data_to_fill.to_csv(data_folder+"1995_09_07_0{i}_TS_DS_F0_VDLC_resnet50_L8FV4PFeb22shuffle1_100000_bx_filtered.csv".format(i=i+1)) 





     