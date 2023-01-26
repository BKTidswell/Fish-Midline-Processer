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

files_3D = os.listdir("3D_Finished_Fish_Data_4P_gaps")
files_2D = os.listdir("Finished_Fish_Data_4P_gaps")

for file_3D in files_3D:

    if file_3D.endswith(".csv"):
        file_id = file_3D[0:22]
        short_id = file_3D[0:10]

        print(file_id)

        #Get the v1 file that matches, and the dlt coefs that go with them both
        file_2D = [f for f in files_2D if file_id in f][0]

        file_3D = "3D_Finished_Fish_Data_4P_gaps/" + file_3D
        file_2D = "Finished_Fish_Data_4P_gaps/" + file_2D

        files_to_do = [file_2D,file_3D]

        for data_file in files_to_do:
            fish_raw_data = pd.read_csv(data_file, index_col=0, header=header)

            df = pd.melt(fish_raw_data)

            #Let's rename things to what they actually are
            #It does it twice since the namming seems different between 2d and 3d
            df = df.rename(columns={"scorer": "Scorer", "individuals": "Fish", "bodyparts": "BodyPart", "coords": "Point"})
            df = df.rename(columns={"variable_0": "Scorer", "variable_1": "Fish", "variable_2": "BodyPart", "variable_3": "Point"})

            #So now I add the Frame to this because I will need to group by that otherwise there is overlap between labels
            df['Frame'] = df.groupby(['Scorer','Fish','BodyPart','Point']).cumcount()+1

            #So now we pivot from the Point column, into seperate ones for x, y, and z
            df = df.pivot_table(index = ['Scorer','Fish','BodyPart','Frame'], columns='Point', values='value')

            #Then we have to reset the index so that we can use 'Scorer','Fish','BodyPart','Frame' as data columns
            df = df.reset_index()

            df = df[df['BodyPart'] == "head"]

            if "3D_DLC" in data_file:
                df_3D = df
                df_3D["Fish"] = df_3D["Fish"] + "_3D"

                max_x_3D = max(df_3D["x"])
                min_x_3D = min(df_3D["x"])
                max_y_3D = max(df_3D["y"])
                min_y_3D = min(df_3D["y"])

                df_3D["y"] = -1 * df_3D["y"] #+ max_y_3D + min_y_3D

                #df_3D["x"] = (df_3D["x"] - min_x_3D)/(max_x_3D - min_x_3D) * (max_x_2D - min_x_2D) + min_x_2D
                #df_3D["y"] = (df_3D["y"] - min_y_3D)/(max_y_3D - min_y_3D) * (max_y_2D - min_y_2D) + min_y_2D

                df_3D["x"] =  df_3D["x"] / 0.07195
                df_3D["y"] =  df_3D["y"] / 0.07195

            else:
                df_2D = df

                max_x_2D = max(df_2D["x"])
                min_x_2D = min(df_2D["x"])
                max_y_2D = max(df_2D["y"])
                min_y_2D = min(df_2D["y"])

                df_2D["x"] =  df_2D["x"] / 193
                df_2D["y"] =  df_2D["y"] / 193

        fig = px.scatter(df_2D, x="x", y="y", color="Fish", opacity = df["Frame"]/max(df["Frame"]))

        fig.add_trace(px.scatter(df_3D, x="x", y="y", color="Fish", opacity = df["Frame"]/max(df["Frame"])).data[0])
        fig.add_trace(px.scatter(df_3D, x="x", y="y", color="Fish", opacity = df["Frame"]/max(df["Frame"])).data[1])
        fig.add_trace(px.scatter(df_3D, x="x", y="y", color="Fish", opacity = df["Frame"]/max(df["Frame"])).data[2])
        fig.add_trace(px.scatter(df_3D, x="x", y="y", color="Fish", opacity = df["Frame"]/max(df["Frame"])).data[3])
        fig.add_trace(px.scatter(df_3D, x="x", y="y", color="Fish", opacity = df["Frame"]/max(df["Frame"])).data[4])
        fig.add_trace(px.scatter(df_3D, x="x", y="y", color="Fish", opacity = df["Frame"]/max(df["Frame"])).data[5])
        fig.add_trace(px.scatter(df_3D, x="x", y="y", color="Fish", opacity = df["Frame"]/max(df["Frame"])).data[6])
        fig.add_trace(px.scatter(df_3D, x="x", y="y", color="Fish", opacity = df["Frame"]/max(df["Frame"])).data[7])

        fig.write_html("2D_3D_Compare_Plots/{name}.html".format(name = file_id), auto_play=False)
