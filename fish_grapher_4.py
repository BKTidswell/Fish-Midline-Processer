import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import interpolate
import numpy.ma as ma
from scipy.interpolate import splprep, splev
from matplotlib.gridspec import GridSpec


n_fish = 5
b_parts_csv = ["head","tail","midline2","midline1","midline3"]
b_parts = ["head","midline1","midline2","midline3","tail",]
fish_colors = ["red","orange","green","blue","purple"]

def DLC_CSV_to_dict(num_fish,fish_parts):
    data_points = ["x","y","prob"]
      
    fish_dict = {}

    for i in range(num_fish):
        fish_dict[i] = {}
        for part in fish_parts:
            fish_dict[i][part] = {}
            for point in data_points:
                fish_dict[i][part][point] = []

    # Give the location of the file 
    file = "N_LLine_A_1_TrimmedDLC_resnet50_Multi_VidsJun3shuffle1_3000_sk_filtered.csv"
      
    # To open Workbook 
    fish_data = pd.read_csv(file)

    cols = fish_data.columns
    time_points = len(fish_data[cols[0]])

    for i in range(0,len(cols)-1):
        fish_num = math.floor(i/15)
        fish_part = fish_parts[math.floor(i/3)%5]
        data_point = data_points[i%3]

        fish_dict[fish_num][fish_part][data_point] = fish_data[cols[i+1]][3:time_points].astype(float).to_numpy()

    return(fish_dict,time_points-3)

def dict_to_fish_time(f_dict,fish_num,time):
    x = np.zeros(n_fish)
    y = np.zeros(n_fish)

    for i in range(n_fish):
        x[i] = f_dict[fish_num][b_parts[i]]["x"][time]
        y[i] = f_dict[fish_num][b_parts[i]]["y"][time]

    return([x,y])


def splprep_predict(x,y,maxTime):

	x = np.asarray(x)
	y = np.asarray(y)
	t = np.asarray(range(maxTime))
	  
	x = ma.masked_where(np.isnan(x), x)
	y = ma.masked_where(np.isnan(y), y)
	  
	if(np.any(x.mask)):
		x = x[~x.mask]
	  
	if(np.any(y.mask)):
		y = y[~y.mask]
	  
	newX = [x[0]]
	newY = [y[0]]
	  
	for i in range(min(len(x),len(y)))[1:]:
		if (abs(x[i] - x[i-1]) > 1e-4) or (abs(y[i] - y[i-1]) > 1e-4):
			newX.append(x[i])
			newY.append(y[i]) 
	      
	newX = np.asarray(newX)
	newY = np.asarray(newY)
	  
	#s is the smoothing
	#I think I might need knots definded

	tck, u = splprep([newX, newY], s=10**4)
	newU = np.arange(0,1,t[1]/(t[-1]+t[1]))
	new_points = splev(newU, tck)
	  
	return(new_points)

def get_dist(p1,p2):
    return(math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2)) )

def get_dot_product(p1,p2):
    return np.dot(p1,p2)

#px2 and py2 are the next points
def get_para_perp_dist(p_fish,p_predict,p_next):

    position_vector = np.asarray([p_predict[0]-p_fish[0],p_predict[1]-p_fish[1],0])
    swim_vector = np.asarray([p_predict[0]-p_next[0],p_predict[1]-p_next[1],0])

    vecDist = math.sqrt(swim_vector[0]**2 + swim_vector[1]**2)

    swim_vector = swim_vector/vecDist

    perp_swim_vector = np.cross(swim_vector,[0,0,1])

    para_coord = np.dot(position_vector,swim_vector)
    perp_coord = np.dot(position_vector,perp_swim_vector)

    return(para_coord,perp_coord)


def generate_midline(one_fish):

    fish_x = one_fish["midline1"]["x"]
    fish_y = one_fish["midline1"]["y"]
    predict_fish = splprep_predict(fish_x,fish_y,time_points)
    
    para_a = []
    perp_a = []

    for i in range(time_points-1):

        temp_para = np.zeros(5)
        temp_perp = np.zeros(5)

        for j in range(len(b_parts)):
            fish_x_b = one_fish[b_parts[j]]["x"]
            fish_y_b = one_fish[b_parts[j]]["y"]

            c_fish = [ fish_x_b[i],fish_y_b[i] ]
            c_point = [ predict_fish[0][i],predict_fish[1][i] ]
            f_point = [ predict_fish[0][i+1],predict_fish[1][i+1] ]
            
            temp_para[j],temp_perp[j] = get_para_perp_dist(c_fish,c_point,f_point)

        para_a.append(temp_para)
        perp_a.append(temp_perp)

    return(para_a,perp_a,predict_fish)


fish_dict,time_points = DLC_CSV_to_dict(num_fish = n_fish, fish_parts = b_parts_csv)

fish_para = []
fish_perp = []
fish_paths = []

for i in range(n_fish):
    f_para_temp,f_perp_temp,estimate_path_temp = generate_midline(fish_dict[i])

    fish_para.append(f_para_temp)
    fish_perp.append(f_perp_temp)
    fish_paths.append(estimate_path_temp)


fig = plt.figure(figsize=(14,7))
axes = []

gs = GridSpec(3, 6, figure=fig, wspace = 0.5, hspace=0.5)
axes.append( fig.add_subplot(gs[0:3, 0:3]) )
axes.append( fig.add_subplot(gs[0, 3]))
axes.append( fig.add_subplot(gs[0, 4]))
axes.append( fig.add_subplot(gs[0, 5]))
axes.append( fig.add_subplot(gs[1, 3]))
axes.append( fig.add_subplot(gs[1, 5]))
axes.append( fig.add_subplot(gs[2, 3]))
axes.append( fig.add_subplot(gs[2, 4]))
axes.append( fig.add_subplot(gs[2, 5]))  

ims=[]

for i in range(time_points-1):
    temp_plots = []

    for j in range(n_fish):
        fish = dict_to_fish_time(fish_dict,j,i)
        bigplot, = axes[0].plot(fish[0], fish[1], color = fish_colors[j], marker='o')

        new_fish_plot, = axes[j+1].plot(fish_para[j][i], fish_perp[j][i], color = fish_colors[j], marker='o')
        new_mid_plot = axes[0].plot(fish_paths[j][0], fish_paths[j][1], color = fish_colors[j])

        temp_plots.append(new_fish_plot)
        temp_plots.append(bigplot)

    ims.append(temp_plots)

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=4000)

#plt.tight_layout()
plt.show()

# x = fish_dict[0][b_parts[3]]["x"]
# y = fish_dict[0][b_parts[3]]["y"]
# predict = splprep_predict(x,y,time_points)

# # plt.figure()
# # plt.plot(x, y,color='blue',linewidth=3)
# # plt.plot(predict[0],predict[1],color='red',linewidth=1)
# # plt.show()

# index = 100
# x = fish_dict[0][b_parts[0]]["x"]
# y = fish_dict[0][b_parts[0]]["y"]

# print(get_para_perp_dist([x[index],y[index]],[predict[0][index],predict[1][index]],[predict[0][index+1],predict[1][index+1]]))


# xParts = []
# yParts = []

# for b in b_parts:
#   x = fish_dict[0][b]["x"]
#   y = fish_dict[0][b]["y"]

#   para,perp = get_perp_para_dist([x[index],y[index]],[predict[0][index],predict[1][index]],[predict[0][index+1],predict[1][index+1]])

#   xParts.append(para)
#   yParts.append(perp)

# plt.plot(xParts, yParts, 'o', color='black')
# plt.show()




# # fig = plt.figure()
# # ax = fig.add_subplot(111)
# # ax.axis([0, 2000, 0, 2000])
# # ims=[]

# fig, ax = plt.subplots(2,3)

# for i in range(n_fish):
#   print(i,math.floor(i/3),i%3)
#   ax[math.floor(i/3),i%3].axis([0, 2000, 0, 2000])

# ims=[]

# for i in range(time_points):
#   temp_plots = []

#   for j in range(n_fish):
#       fish = dict_to_fish_time(fish_dict,j,i)
#       temp_plots.append(ax[math.floor(j/3),j%3].scatter(fish[0], fish[1], s=1, color= fish_colors[j]))

#   ims.append(temp_plots)

# ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,repeat_delay=2000)

# plt.show()

