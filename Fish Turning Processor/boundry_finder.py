import matplotlib.pyplot as plt
import pandas
import os
import numpy as np

xs = []
ys = []
videos = []

folder = "Eight_Fish_Data/"

header = list(range(4))

fish_names = ["individual1","individual2",
              "individual3","individual4",
              "individual5","individual6",
              "individual7","individual8"]

for file_name in os.listdir(folder):
    if file_name.endswith(".csv"):
        print(file_name)

        fish_data = pandas.read_csv(folder+file_name,index_col=0, header=header)

        year = file_name[0:4]
        month = file_name[5:7]
        day = file_name[8:10]
        trial = file_name[11:13]
        abalation = file_name[15:16]
        darkness = file_name[18:19]
        flow = file_name[21:22]

        scorerer = fish_data.keys()[0][0]

        for fish in fish_names:
            head_x_data = fish_data[scorerer][fish]["head"]["x"].to_numpy()
            head_y_data = fish_data[scorerer][fish]["head"]["y"].to_numpy()

            xs = np.append(xs,head_x_data)
            ys = np.append(ys,head_y_data)
            videos = np.append(videos,np.repeat(file_name[0:10],len(head_x_data)))

labels, index = np.unique(videos, return_inverse=True)
        
fig, ax = plt.subplots()
sc = ax.scatter(xs, ys, marker = 'o', c = index, alpha = 0.8)
ax.legend(sc.legend_elements()[0], labels)
plt.show()
