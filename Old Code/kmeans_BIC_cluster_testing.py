
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = X.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)



# # IRIS DATA
# iris = sklearn.datasets.load_iris()
# X = iris.data[:, :4]  # extract only the features
# #Xs = StandardScaler().fit_transform(X)
# Y = iris.target

points = np.asarray([(1389.989, 479.734), (179.192, 643.001), (830.585, 498.093), (288.339, 445.576), (1477.155, 457.5), (1731.497, 383.914), (1762.415, 216.911)])

points = np.asarray([(575.532, 442.825), (1286.635, 357.27), (254.613, 439.522), (630.225, 573.112), (406.878, 130.639), (1844.596, 219.098), (1728.061, 328.715), (1575.037, 305.029)])

# xs = [100, 120, 100, 120, 0, 20, 0, 20]
# ys = [100, 120, 120, 100, 0, 20, 20, 0]

# points = np.asarray([item for item in zip(xs, ys)])

ks = range(1,len(points))

# run 9 times kmeans and save each result in the KMeans object
KMeans = [cluster.KMeans(n_clusters = i, init="k-means++").fit(points) for i in ks]

# now run for each cluster the BIC computation
BIC = [compute_bic(kmeansi,points) for kmeansi in KMeans]

print(BIC)

# def get_km(k, X):
#     km = KMeans(n_clusters=k, random_state=37)
#     km.fit(X)
#     return km

# def get_aic(k, X):
#     gmm = GaussianMixture(n_components=k, init_params='kmeans')
#     gmm.fit(X)
#     return gmm.aic(X)

# for k in range(1,9):
#     aic = get_aic(k, points)
#     print(aic)

n_clusters = np.argmax(BIC)+1
print(n_clusters)

km_final = cluster.KMeans(n_clusters = n_clusters, init="k-means++").fit(points)

labels = km_final.labels_
cluster_centers_ = km_final.cluster_centers_


w0 = points[labels == 0]
w1 = points[labels == 1]
w2 = points[labels == 2]
plt.plot(w0[:, 0], w0[:, 1], 'o', alpha=0.5, label='cluster 0')
plt.plot(w1[:, 0], w1[:, 1], 'd', alpha=0.5, label='cluster 1')
plt.plot(w2[:, 0], w2[:, 1], 's', alpha=0.5, label='cluster 2')
plt.plot(cluster_centers_[:, 0], cluster_centers_[:, 1], 'k*', label='centroids')
plt.axis('equal')
plt.legend(shadow=True)
plt.show()



