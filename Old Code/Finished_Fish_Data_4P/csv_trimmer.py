import csv
import os
import numpy as np
import h5py
import pandas as pd


unwanted_body_parts = ["midline1","midline3"]

top_dir = "/Users/Ben/Desktop/Fish Midline Processer/Finished_Fish_Data_4P/"
all_folders = os.walk(top_dir, topdown=True, onerror=None, followlinks=False)

fname = []

for root,d_names,f_names in all_folders:
    for f in f_names:
        if f.endswith("filtered.csv"):
            fname.append(os.path.join(root, f))

for filename in fname:

    wanted_cols = []

    old_filename = filename.replace("filtered","filtered_Old")
    #h5_filename = filename.replace("csv","h5")

    #Change name of old file so the new file can be good
    os.rename(filename,old_filename)

    #This is to just get the bodyparts we want and nothing else
    with open(old_filename, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for i,row in enumerate(csv_reader):

            if i == 2:
                wanted_cols = np.asarray([r not in unwanted_body_parts for r in row])
                break

    #Then read the csv and write only the columns we want by removing the other ones
    with open(old_filename, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)

        with open(filename,"w") as write_obj:
            csv_writer = csv.writer(write_obj)

            for row in csv_reader:
                np_row = np.asarray(row)
                csv_writer.writerow(np_row[wanted_cols])

    #Get rid of old file
    os.remove(old_filename)




            
