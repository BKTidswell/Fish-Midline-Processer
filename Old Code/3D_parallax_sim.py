
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import math
import pandas as pd

np.set_printoptions(suppress=True)

# instrisic_matrix = np.array([[2800, 0, 0, 0],
#                              [0, 2800, 0, 0],
#                              [0, 0, 1, 0]])

# point_matrix = np.array([[0],
#                          [0],
#                          [2800],
#                          [1]])


# w = (instrisic_matrix @ point_matrix)[2][0]

# final_matrix = (instrisic_matrix @ point_matrix)/w

# print(final_matrix)

#Okay so at 2800 if I give it x and y values it gives them back to me
# Which means that a distance of 2800... pixels? (God what are the units) away is where the camera plane is???
# I think?
# I want to double check this, but I could then look 200 beyond that? I think the units are right.

def getDist(x1s,y1s,x2s,y2s):
    dist = np.sqrt((x1s-x2s)**2+(y1s-y2s)**2)
    return dist


def getUV(x,y,z):
    instrisic_matrix = np.array([[2800, 0, 0, 0],
                             [0, 2800, 0, 0],
                             [0, 0, 1, 0]])

    point_matrix = np.array([[x],
                             [y],
                             [z],
                             [1]])


    w = (instrisic_matrix @ point_matrix)[2][0]

    final_matrix = (instrisic_matrix @ point_matrix)/w

    return final_matrix[0][0], final_matrix[1][0]


print(getUV(100,200,3000))

final_array = np.zeros((11,11))



xVals = np.arange(-1280, 1281, 50)
yVals = np.arange(-500, 501, 50)

final_array = np.zeros((len(xVals),len(yVals)))

graph_xs = np.zeros(len(xVals)*len(yVals))
graph_ys = np.zeros(len(xVals)*len(yVals))
graph_values = np.zeros(len(xVals)*len(yVals))

for i, xVal in enumerate(xVals):
    for j, yVal in enumerate(yVals):
        
        u1, v1 = getUV(xVal,yVal,2800)
        u2, v2 = getUV(xVal,yVal,3000)

        distance = getDist(u1,v1,u2,v2)

        final_array[i][j] = distance

        graph_xs[i*len(yVals)+j] = xVal
        graph_ys[i*len(yVals)+j] = yVal
        graph_values[i*len(yVals)+j] = distance

#print(final_array)

my_pd = pd.DataFrame(data = np.asarray([graph_xs,graph_ys,graph_values]).T , columns=['X','Y','Distance'])

my_pd.to_csv("Parallax_Distances.csv",index=False)

plt.scatter(graph_xs, graph_ys, s=100, c=graph_values, alpha=1)
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.colorbar()
plt.show()







