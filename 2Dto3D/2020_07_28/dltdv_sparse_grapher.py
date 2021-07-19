import numpy as np
import pandas as pd
import math
import plotly.express as px

n_fish = 8
b_parts_csv = ["head","tailbase","midline2","tailtip"]
n_b_parts = len(b_parts_csv)
b_parts = ["head","midline2","tailbase","tailtip"]
poses = ["x","y","z"]

sparse_csv = pd.read_csv("DLTdv8_data_xyzpts.csv")

plotly_data = {"Fish":[], "BodyPart":[], "x":[], "y":[], "z":[], "Frame":[]}

for index, row in sparse_csv.iterrows():

    pos = poses[int((row['Num']-1)%3)]
    point = math.floor((row['Num']-1)/3)
    b_part = b_parts[int(point%n_b_parts)]
    fish = math.floor(point/4)

    if pos == "x":
        plotly_data["Fish"].append(fish)
        plotly_data["BodyPart"].append(b_part)
        plotly_data[pos].append(row['Pos'])
        plotly_data["Frame"].append(row['Frame'])

    else:
        plotly_data[pos].append(row['Pos'])

df = pd.DataFrame(plotly_data)
fig = px.scatter_3d(df,x="x", y="y", z="z", color="Fish", animation_frame="Frame", hover_data = ["BodyPart"])
fig.show()
