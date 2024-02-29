from clustering import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def show_clusters(method, n_clusters, rank, approx):
    df, fvs = cluster(method, n_clusters, 'rows', rank, approx)
    labels = df["Cluster"].to_list()
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(fvs)

    clusterXs = dict()
    clusterYs = dict()
    for i in range(n_clusters):
        clusterXs[i] = []
        clusterYs[i] = []
    for i in range(len(labels)):
        clusterXs[labels[i]].append(pcs[i, 0])
        clusterYs[labels[i]].append(pcs[i, 1])

    colors = ["y", "r", "g", "brown", "b", "c", "m", "purple", "wheat", "aqua"]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    ax1.set_title("Clustering met " + method + " met rang " + str(rank))
    fig.patch.set_facecolor('#DEEBF7')
    ax1.set_facecolor('#DEEBF7')
    for i in range(n_clusters):
        ax1.scatter(clusterXs[i], clusterYs[i], c=colors[i])
    plt.show()


show_clusters("vector_aca_t", 3, 10, 3)
# show_clusters("cp", 3)
