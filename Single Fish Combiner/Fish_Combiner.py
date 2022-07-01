import pandas as pd
import numpy as np
from random import *
import os

#Header list for reading the raw location CSVs
header_3 = list(range(3))
header_4 = list(range(4))

data_folders = ["Good Single Fish/light/still/","Good Single Fish/light/flow/",
                "Good Single Fish/dark/still/","Good Single Fish/dark/flow/",
                "Good Single Fish/ablation/still/","Good Single Fish/ablation/flow/"]

for data_folder in data_folders:

    if os.path.exists(data_folder+".DS_store"):
      os.remove(data_folder+".DS_store")
    else:
      print("No DS Store") 

    data_files = os.listdir(data_folder)

    data_to_fill = pd.read_csv("dummy_data.csv", index_col=0, header=header_4)

    # Make 10 combo files

    # The smallest number of data files for a condition is 13, which still gives us 1287 combinations for that 
    for x in range(10):
        shuffle(data_files)
        all_data_arr = [pd.read_csv(data_folder+f, index_col=0, header=header_3) for f in data_files]

        date = data_files[0][0:10]
        conds = data_files[0][14:22]

        #Store and reset the file numbers
        file_numbers = []

        # Get 8 files to fill it in with
        for i in range(8):
            file_numbers.append(data_files[i][11:13])

            data_to_fill.iloc[: ,i*12:(i+1)*12] = all_data_arr[i].iloc[: ,0:13]

        # Sort the numbers first to make sure we don't get any duplicates
        file_numbers.sort()

        data_to_fill.to_csv("Multi Data/{date}_{trial:>02d}_{cond}_V1_{files}.csv".format(date = date, cond = conds, trial = x+1, i=i+1, files = "_".join(file_numbers))) 






