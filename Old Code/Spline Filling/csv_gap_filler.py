import os, sys
import itertools
import h5py
import numpy as np
import pandas as pd
from deeplabcut.post_processing import columnwise_spline_interp


gap_folder_path = "/Users/Ben/Desktop/Fish Midline Processer/Finished_Fish_Data_4P_gaps/"
filled_folder_path = "/Users/Ben/Desktop/Fish Midline Processer/Finished_Fish_Data_4P_gaps_filled/"

video_files = os.listdir(gap_folder_path)
n_fish = 8

individuals = ["individual" + str(i+1) for i in range(n_fish)]
bodyparts = ["head","midline2","tailbase","tailtip"]
coords = ["x","y","likelihood"]

for file in video_files:
    if file.endswith(".csv"):
        #df = pd.read_hdf(gap_folder_path+file)
        df = pd.read_csv(gap_folder_path+file, index_col=0, header = [0,3])
        scorer = [df.columns[0]]

        tuples = list(itertools.product(scorer,individuals,bodyparts,coords))
        index = pd.MultiIndex.from_tuples(tuples, names=["scorer", "individuals", "bodyparts", "coords"])
        df.columns = index

        data = df.to_numpy()
        mask = ~df.columns.get_level_values(level="coords").str.contains("likelihood")
        xy = data[:, mask]
        prob = data[:, ~mask]
        missing = np.isnan(xy)
        xy_filled = columnwise_spline_interp(xy, 0)
        filled = ~np.isnan(xy_filled)
        xy[filled] = xy_filled[filled]
        inds = np.argwhere(missing & filled)
        if inds.size:
            # Retrieve original individual label indices
            inds[:, 1] //= 2
            inds = np.unique(inds, axis=0)
            prob[inds[:, 0], inds[:, 1]] = 0.01
        data[:, mask] = xy
        data[:, ~mask] = prob
        df = pd.DataFrame(data, index=df.index, columns=df.columns)

        df.to_csv(filled_folder_path+file.replace(".csv","_filled.csv"), mode="w")
        df.to_hdf(filled_folder_path+file.replace(".csv",".h5"), "df_with_missing", format="table", mode="w")
