import numpy as np
import pandas as pd
import math
import plotly.express as px
from scipy import linalg

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

#This only does one point at a time!!!
def DLTrecon(Ls, uvs):
	'''
	From: https://github.com/Fishified/Tracker3D/blob/master/DLT.py
	Reconstruction of object point from image point(s) based on the DLT parameters.
	This code performs 2D or 3D DLT point reconstruction with any number of views (cameras).
	For 3D DLT, at least two views (cameras) are necessary.
	Inputs:
	 nd is the number of dimensions of the object space: 3 for 3D DLT and 2 for 2D DLT. (for me nd is always 3)
	 nc is the number of cameras (views) used.                                          (for me nc is always 2)
	 Ls (array type) are the camera calibration parameters of each camera 
		(is the output of DLTcalib function). The Ls parameters are given as columns
		and the Ls for different cameras as rows.
	 uvs are the coordinates of the point in the image 2D space of each camera.
		The coordinates of the point are given as columns and the different views as rows.
	Outputs:
	 xyz: point coordinates in space
	'''
	
	nc = 2

	#Convert Ls to array:
	Ls = np.asarray(Ls)  
	M = []
	for i in range(nc):
			L = Ls[i,:]
			u,v = uvs[i][0], uvs[i][1] #this indexing works for both list and numpy array
			M.append( [L[0]-u*L[8], L[1]-u*L[9], L[2]-u*L[10], L[3]-u*L[11]] )
			M.append( [L[4]-v*L[8], L[5]-v*L[9], L[6]-v*L[10], L[7]-v*L[11]] )
	
	#Find the xyz coordinates:
	U, S, Vh = np.linalg.svd(np.asarray(M))
	#Point coordinates in space:
	xyz = Vh[-1,0:-1] / Vh[-1,-1]
	
	return xyz

def DLTdvRecon(Ls, uvs):
	Ls = np.array(Ls)
	uvs = np.array(uvs)

	#http://kwon3d.com/theory/dlt/dlt.html#3d

	#https://github.com/tlhedrick/dltdv/blob/master/DLTdv8a_internal/dlt_reconstruct_v2.m
	#Added the extra 2 at the end since numpy needs you to tell it that two things are
	# going there ahead of time

	#Okay so this is a bit of a mess but if each of the right hands sides of the equals signs
	# gives a value pair of [1,2] then [3,4] etc. the matlab matrixes look like this:

	# m1
	#  1     3     5
	#  7     9    11
	#  2     4     6
	#  8    10    12

	# m2
	#  1
	#  3
	#  2
	#  4

	#Actually this is wrong, it doesn't use two points, it just uses the x and y from the second camera 
	# and only the Ls from the second camera? I think?

	# 4/29 Actually actually this is right, and I was wrong about it since cdx in the matlab
	#  code is an array [1,2] and so gets two values from uvs and Ls

	# print(Ls.T)
	# print(uvs)

	# print(Ls[:,0])
	# print(uvs[:,0])

	m1 = np.zeros((4,3))
	m2 = np.zeros((4,1))

	#Alright so now let's slice weirdly

	#Use both sets of points and cameras. cdx is an array

	#print(np.multiply(uvs[:,0], Ls[:,8]) - Ls[:,0])

	m1[0:3:2,0] = uvs[:,0] * Ls[:,8] - Ls[:,0]
	m1[0:3:2,1] = uvs[:,0] * Ls[:,9] - Ls[:,1]
	m1[0:3:2,2] = uvs[:,0] * Ls[:,10] - Ls[:,2]

	m1[1:4:2,0] = uvs[:,1] * Ls[:,8] - Ls[:,4]
	m1[1:4:2,1] = uvs[:,1] * Ls[:,9] - Ls[:,5]
	m1[1:4:2,2] = uvs[:,1] * Ls[:,10] - Ls[:,6]

	m2[0:3:2,0] = Ls[:,3] - uvs[:,0]
	m2[1:4:2,0] = Ls[:,7] - uvs[:,1]

	# m1[0,0] = np.multiply(uvs[1][0],Ls[1][8]) - Ls[1][0]
	# m1[0,1] = np.multiply(uvs[1][0],Ls[1][9]) - Ls[1][1]
	# m1[0,2] = np.multiply(uvs[1][0],Ls[1][10]) - Ls[1][2]

	# m1[1,0] = np.multiply(uvs[1][1],Ls[1][8]) - Ls[1][4]
	# m1[1,1] = np.multiply(uvs[1][1],Ls[1][9]) - Ls[1][5]
	# m1[1,2] = np.multiply(uvs[1][1],Ls[1][10]) - Ls[1][6]

	# m2[0,0] = Ls[1][3] - uvs[1][0]
	# m2[1,0] = Ls[1][7] - uvs[1][1]

	# print("\nm1 and m2")
	# print(m1)
	# print(m2)
	# print()

	#https://medium.com/italiandirectory-publishing/linear-equations-with-python-the-qr-decomposition-66a48b8be89d

	#Nothing seems to work since A isn't square??? Should it be square???

	# Q, R = np.linalg.qr(m1,mode="complete") # QR decomposition with qr function 
	# y = np.dot(Q.T, m2) # Let y=Q'.B using matrix multiplication 
	# xyz = np.linalg.solve(R, y) # Solve Rx=y 
	xyz = linalg.lstsq(m1, m2, lapack_driver = "gelsy")[0]
	#print(xyz)

	return xyz


n_fish = 8
b_parts_csv = ["head","tailbase","midline2","tailtip"]
n_b_parts = len(b_parts_csv)
b_parts = ["head","midline2","tailbase","tailtip"]

fish_dict_V1,time_points_V1 = DLC_CSV_to_dict(num_fish = n_fish, fish_parts = b_parts_csv, file = "2020_07_28_11_TN_DN_F0_V1DLC_resnet50_L8FVJul4shuffle1_100000_bx_filtered.csv")
fish_dict_V2,time_points_V2 = DLC_CSV_to_dict(num_fish = n_fish, fish_parts = b_parts_csv, file = "2020_07_28_11_TN_DN_F0_V2DLC_resnet50_L8FV4PFeb22shuffle1_100000_bx_filtered.csv")

calib_csv = pd.read_csv("4_22_21_easy_wand_calib_dltCoefs.csv")
calib_cols = calib_csv.columns

camera_coeffs = np.zeros((2,11))
camera_coeffs[0] = np.array(calib_csv["C1"])
camera_coeffs[1] = np.array(calib_csv["C2"])

f1_head_points_V1 = np.zeros((time_points_V1,2))
f1_head_points_V2 = np.zeros((time_points_V2,2))

for i,p in enumerate(["x","y"]):
	f1_head_points_V1[:,i] = fish_dict_V1[0]["head"][p]
	f1_head_points_V2[:,i] = fish_dict_V2[0]["head"][p]

f1_head_points_V1V2 = np.asarray([f1_head_points_V1[0],f1_head_points_V2[0]])

data_points = ["x","y","z"]
		
fish_dict_3D = {}

#Set up the 3D Dict
for i in range(n_fish):
	fish_dict_3D[i] = {}
	for part in b_parts:
		fish_dict_3D[i][part] = {}
		for point in data_points:
			fish_dict_3D[i][part][point] = np.zeros(time_points_V1)

for fish in range(n_fish):
	for part in b_parts:
		for i in range(time_points_V1):
			points_3D = DLTdvRecon(camera_coeffs,[[fish_dict_V1[fish][part]["x"][i],fish_dict_V1[fish][part]["y"][i]] , [fish_dict_V2[fish][part]["x"][i],fish_dict_V2[fish][part]["y"][i]]])

			fish_dict_3D[fish][part]["x"][i] = points_3D[0]
			fish_dict_3D[fish][part]["y"][i] = points_3D[1]
			fish_dict_3D[fish][part]["z"][i] = points_3D[2]

#Okay so for plotly we want a n_fish*n_b_parts by 5 (fish,bp,x,y,z) by timepoints array

plotly_data = {"Fish":[], "BodyPart":[], "x":[], "y":[], "z":[], "Frame":[]}

for t in range(time_points_V1):
	for f in range(n_fish):
		for p,part in enumerate(b_parts): 

			plotly_data["Fish"].append(f)
			plotly_data["BodyPart"].append(part)
			plotly_data["x"].append(fish_dict_3D[f][part]["x"][t])
			plotly_data["y"].append(fish_dict_3D[f][part]["y"][t])
			plotly_data["z"].append(fish_dict_3D[f][part]["z"][t])
			plotly_data["Frame"].append(t)

df = pd.DataFrame(plotly_data)

fig = px.scatter_3d(df,x="x", y="y", z="z", color="Fish", animation_frame="Frame", hover_data = ["BodyPart"])

xmin = df["x"].min()
xmax = df["x"].max()
ymin = df["y"].min()
ymax = df["y"].max()
zmin = df["z"].min()
zmax = df["z"].max()

print(df["x"][0])
print(df["y"][0])
print(df["z"][0])
print(df["Fish"][0])
print(df["BodyPart"][0])
print(df["Frame"][0])

# fig.update_layout(
# 	scene = dict(
# 		xaxis = dict(nticks=4, range=[-0.6,0.6],),
# 		yaxis = dict(nticks=4, range=[-0.2,0.2],),
# 		zaxis = dict(nticks=4, range=[-0.3,0.3],),))

fig.show()


