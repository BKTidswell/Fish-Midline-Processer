# Requires aniposelib. Assuming you already have Anaconda python set up, you can install aniposelib with

#     % python -m pip install aniposelib

# Also requires the Python interface to OpenCV. This should install as part of your normal Anaconda installation, but if it didn't, you can use

#     % conda install opencv

# Everything else should be standard Python install

import os
import aniposelib
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Set up to detect a CHaRuCO board. Make sure you identify the number of squares and their size in mm. 
# The numbers below should be correct for the board we have in the lab.

# marker_length, marker_bits, and dict_size are particular to the sort of CHaRuCO board that we have in the lab. 
# Don't change them unless you know what you're doing.

board = aniposelib.boards.CharucoBoard(squaresX=6, squaresY=6,
                                        square_length=24.33, marker_length=17, marker_bits=5, dict_size=50)

#Fill in the path and the names of video files that have the CHaRuCO images. 
# For paths on Windows machines, be careful not to remove the r at the beginning of the quote; 
#  without it, all backslashes will be registered as special characters.

videopath = r'/Users/Ben/Desktop/Charuco_DLT/charuco_boards/2023_02_09'
videonames = ['checkerboard_V1.mp4','checkerboard_V2.mp4']

#This will detect points in the first video, just for testing purposes.

video1 = os.path.join(videopath, videonames[0])
rows = board.detect_video(video1, progress=True)

print(len(rows))

# Check detected corners
# This will load in a frame from the video and show the detected corners.

cap = cv2.VideoCapture(video1)

#You can change it to look at a different frame

i = 1000
fr = rows[i]['framenum']

cap.set(1, fr)
ret, frame = cap.read()

fig, ax = plt.subplots()
ax.imshow(frame)
ax.plot(rows[i]['corners'][:,0,0], rows[i]['corners'][:,0,1], 'ro')
plt.show()

#Detect all corners in video
camdf = []

for camnum, video1 in enumerate(videonames):
    print(video1)
    fn = os.path.join(videopath, video1)

    rows = board.detect_video(fn, progress=True)

    for row1 in rows:
        df1 = pd.DataFrame(data={'id': row1['ids'][:,0], 'x': row1['corners'][:,0,0], 'y': row1['corners'][:,0,1]})
        df1['frame'] = row1['framenum']
        df1['camera'] = camnum
        camdf.append(df1)

#Merge the data frames.
dfall = pd.concat(camdf, ignore_index=True)

#Look at the beginning of the data set.
print(dfall.head())

#And at the end. You should see x y points for multiple frames and multiple cameras.
dfall.tail()

#And save them all to CSV.
dfall.to_csv(os.path.join(videopath, 'boards.csv'))
#Next step: Run the R notebook rearrange_point_for_easywand.Rmd to rearrange the points in the CSV file so that EasyWand can load them in.

