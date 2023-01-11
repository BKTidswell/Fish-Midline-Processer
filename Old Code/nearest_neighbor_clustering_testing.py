#Okay so I feel like the issue with the previous stuff was that it didn't take the scale of the school into account
# like if all the fish are close but in two clusters I would probably call it one school, even though kmeans would not
# Not sure I am explaining this the best.
#But basically what I really care about is how close the fish are, and if that forms two schools
# So I was thinking to look within 2 BL and make like... a node graph? We'll see

from scipy.spatial import distance_matrix
import pylab
import numpy as np
import networkx as nx
from networkx.algorithms import community

points = np.asarray([(1389.989, 479.734), (179.192, 643.001), (830.585, 498.093), (288.339, 445.576), (1477.155, 457.5), (1731.497, 383.914), (1762.415, 216.911)])

#points = np.asarray([(575.532, 442.825), (1286.635, 357.27), (254.613, 439.522), (630.225, 573.112), (406.878, 130.639), (1844.596, 219.098), (1728.061, 328.715), (1575.037, 305.029)])

# xs = [100, 120, 100, 120, 0, 20, 0, 20]
# ys = [100, 120, 120, 100, 0, 20, 20, 0]

# points = np.asarray([item for item in zip(xs, ys)])*4

#Alright so this calclates all this distances and sets them to false if they are alobe 2
# Then we make a network graph and find all the connected componenets in there, tells us how many groups there are

fish_len = 193
min_BL = 2

points = points/fish_len

dm = distance_matrix(points,points)
dm_min = dm <= min_BL

# print(dm)
# print(dm_min)

G = nx.from_numpy_array(dm_min)
#print(G.edges(data=True))

print(len([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]))

nx.draw(G, with_labels=True)
pylab.show()
