import numpy as np
from helpers import k_means


def cluster(n_clusters, dir):
    tensor = np.load("saved_tensors/full_tensor.npy")
    k, i, j = tensor.shape[0], tensor.shape[1], tensor.shape[2]
    print(k,i,j)
    centers, labels = k_means(tensor.reshape(k*i, j), n_clusters)
    print(tensor.reshape(k*i, j).shape)
    print(labels)


np.set_printoptions(threshold=np.inf)
cluster(3, 0)

