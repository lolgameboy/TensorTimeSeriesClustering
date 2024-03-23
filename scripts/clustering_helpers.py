import numpy as np
import pandas as pd

from helpers import k_means
from tensorly.decomposition import parafac
import tensor as t
from sklearn.metrics.cluster import adjusted_rand_score

from scripts.data_class import DAL
from vector_aca_t import *


def cluster(method, n_clusters, direction, max_rank, max_approx=0):
    tensor = np.load("saved_tensors/full_tensor.npy")
    if method == "cp":
        cp_facts = get_CP_decomposition(tensor, max_rank)[1]
        if direction == 'rows':
            feature_vectors = cp_facts[2]
        elif direction == 'columns':
            feature_vectors = cp_facts[1]
        elif direction == 'tubes':
            feature_vectors = cp_facts[0]
    elif method == "vector_aca_t":
        decomp = vector_aca_t(tensor, max_rank, max_approx)
        if direction == 'rows':
            feature_vectors = decomp.get_row_vectors().transpose()
        elif direction == 'columns':
            feature_vectors = decomp.get_column_vectors().transpose()
        elif direction == 'tubes':
            feature_vectors = decomp.get_tube_vectors().transpose()
    else:
        return

    people, exercises, sensors = t.get_people_exercises_sensors()

    centers, labels = k_means(feature_vectors, n_clusters=n_clusters)
    if direction == 'tubes':
        data = {"Sensor": sensors, "Cluster": labels}
    else:
        data = {"Person": people, "Exercise": exercises, "Cluster": labels}
    dataframe = pd.DataFrame(data=data)

    return dataframe, feature_vectors


def get_CP_decomposition(tensor, max_rank):
    factors = parafac(tensor, rank=max_rank, normalize_factors=False)
    return factors


def get_overview():
    dal = DAL("amie-kinect-data.hdf")
    overview = dal.overview()
    df = pd.DataFrame(overview)
    return df


np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_rows", 1000)

# print(get_overview())

frame, fvs = cluster("vector_aca_t", 5, 'rows', 25, 3)
print(frame)
frame, fvs = cluster("cp", 5, 'rows', 25, 3)
print(frame)

# tensor = np.load("saved_tensors/full_tensor.npy")
# cp_facts = get_CP_decomposition(tensor, 10)[1]
# feature_vectors = cp_facts[2]
# print(feature_vectors.shape)
# print(feature_vectors)
