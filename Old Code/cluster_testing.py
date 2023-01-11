

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
from sklearn.cluster import AffinityPropagation, DBSCAN


xs = [100, 120, 100, 120, 0, 20, 0, 20]
ys = [100, 120, 120, 100, 0, 20, 20, 0]

points = [[xs[i], ys[i]] for i in range(len(xs))]

points = [item for item in zip(xs, ys)]

points = np.asarray([(1389.989, 479.734), (179.192, 643.001), (830.585, 498.093), (288.339, 445.576), (1477.155, 457.5), (1731.497, 383.914), (1762.415, 216.911)])

print(points)

points = np.asarray([(784.209, 651.65), (1366.786, 442.011), (140.535, 643.57), (743.679, 569.116), (335.934, 292.671), (1690.066, 381.606), (1758.071, 259.366)])

points = np.asarray([(1175.52, 120.527), (1479.724, 430.816), (804.768, 181.006), (920.478, 344.87), (697.173, 616.645), (856.145, 440.639), (1632.632, 515.509)])

#points = np.asarray([[100,100], [120,120], [100,120], [120,100], [0,0], [20,20], [0,20], [20,0]])

#plt.scatter(points[:,0], points[:,1], cmap='viridis')
#plt.show()

# create kmeans object
# kmeans = DBSCAN()# fit kmeans object to data
# kmeans.fit(points)

# print(kmeans.labels_)

# labels = kmeans.labels_

# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# print(n_clusters_)

# print location of clusters learned by kmeans object
#print(kmeans.cluster_centers_)# save new clusters for chart

#whitened = whiten(points)
# codes = 7

# print(kmeans2(whitened,codes))

# print(whitened)

# centroid, label = kmeans2(whitened, 1, minit='points')
# print(centroid, label)

# w0 = whitened[label == 0]
# w1 = whitened[label == 1]
# w2 = whitened[label == 2]
# plt.plot(w0[:, 0], w0[:, 1], 'o', alpha=0.5, label='cluster 0')
# plt.plot(w1[:, 0], w1[:, 1], 'd', alpha=0.5, label='cluster 1')
# plt.plot(w2[:, 0], w2[:, 1], 's', alpha=0.5, label='cluster 2')
# plt.plot(centroid[:, 0], centroid[:, 1], 'k*', label='centroids')
# plt.axis('equal')
# plt.legend(shadow=True)
# plt.show()


af = AffinityPropagation().fit(points)

cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)
cluster_centers_ = af.cluster_centers_


print(n_clusters_)
print(af.labels_)
print(cluster_centers_indices)

w0 = points[labels == 0]
w1 = points[labels == 1]
w2 = points[labels == 2]
plt.plot(w0[:, 0], w0[:, 1], 'o', alpha=0.5, label='cluster 0')
plt.plot(w1[:, 0], w1[:, 1], 'd', alpha=0.5, label='cluster 1')
plt.plot(w2[:, 0], w2[:, 1], 's', alpha=0.5, label='cluster 2')
plt.plot(cluster_centers_[:, 0], cluster_centers_[:, 1], 'k*', label='centroids')
plt.axis('equal')
plt.legend(shadow=True)

#Kinda defines the school
plt.xlim([0, 2000])
plt.ylim([0, 1000])
plt.show()


