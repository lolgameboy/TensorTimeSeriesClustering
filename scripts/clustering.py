import numpy as np
import pandas as pd

from helpers import k_means
from tensorly.decomposition import parafac
import tensor as t
from sklearn.metrics.cluster import adjusted_rand_score


def cluster(n_clusters, feature_vect):
    tensor = np.load("saved_tensors/full_tensor.npy")

    cp_facts = get_CP_decomposition(tensor, 50)[1]
    if feature_vect == 'rows':
        to_cluster = cp_facts[2]
    else:
        to_cluster = cp_facts[0]

    people, exercises, sensors = t.get_people_exercises_sensors()

    centers, labels = k_means(to_cluster, n_clusters=n_clusters)
    data = {"Person": people, "Exercise": exercises, "Cluster": labels}
    dataframe = pd.DataFrame(data=data)
    return dataframe


def get_CP_decomposition(tensor, max_rank):
    factors = parafac(tensor, rank=max_rank, normalize_factors=False)
    return factors


np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_rows", 1000)
print(cluster(3, 'rows'))
