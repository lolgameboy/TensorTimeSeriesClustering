import numpy as np
import pandas as pd

from helpers import k_means
from tensorly.decomposition import parafac
import tensor as t
from sklearn.metrics.cluster import adjusted_rand_score
from vector_aca_t import *


def cluster(method, n_clusters, direction, max_rank, max_approx=0):
    tensor = np.load("saved_tensors/full_tensor.npy")
    if method == "cp":
        cp_facts = get_CP_decomposition(tensor, max_rank)[1]
        feature_vectors = cp_facts[2]
    elif method == "vector_aca_t":
        decomp = vector_aca_t(tensor, max_rank, max_approx)
        feature_vectors = decomp.get_row_vectors().transpose()
    else:
        return

    # if direction == 'rows':
    #     feature_vectors = cp_facts[2]
    # else:
    #     feature_vectors = cp_facts[0]

    people, exercises, sensors = t.get_people_exercises_sensors()

    centers, labels = k_means(feature_vectors, n_clusters=n_clusters)
    data = {"Person": people, "Exercise": exercises, "Cluster": labels}
    dataframe = pd.DataFrame(data=data)

    return dataframe, feature_vectors


def get_CP_decomposition(tensor, max_rank):
    factors = parafac(tensor, rank=max_rank, normalize_factors=False)
    return factors


np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_rows", 1000)
# print(cluster(3, 'rows'))

# tensor = np.load("saved_tensors/full_tensor.npy")
# cp_facts = get_CP_decomposition(tensor, 10)[1]
# feature_vectors = cp_facts[2]
# print(feature_vectors.shape)
# print(feature_vectors)
