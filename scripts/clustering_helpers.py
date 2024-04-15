import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from tensorly.decomposition import parafac
import tensor as t
from sklearn.metrics.cluster import adjusted_rand_score

from data_class import DAL
from vector_aca_t import *


def cluster(method, n_clusters, direction, max_feature_vectors, max_approx=1):
    tensor = np.load("../saved_tensors/full_tensor.npy")
    if method == "cp":
        feature_vectors = get_CP_factors(direction, max_feature_vectors)
    elif method == "vector_aca_t":
        max_rank = (max_feature_vectors // max_approx) + 1
        decomp = vector_aca_t(tensor, max_rank, max_approx)
        if direction == 'rows':
            feature_vectors = decomp.get_row_vectors().transpose()
        elif direction == 'columns':
            feature_vectors = decomp.get_column_vectors().transpose()
        elif direction == 'tubes':
            feature_vectors = decomp.get_tube_vectors().transpose()
        feature_vectors = feature_vectors[:,0:max_feature_vectors]
    else:
        return

    centers, labels = k_means(feature_vectors, n_clusters=n_clusters)
    return labels


def cluster_table(method, n_clusters, direction, max_rank, max_approx=0):
    labels = cluster(method, n_clusters, direction, max_rank, max_approx)

    people, exercises, sensors = t.get_people_exercises_sensors()
    if direction == 'tubes':
        data = {"Sensor": sensors, "Cluster": labels}
    else:
        data = {"Person": people, "Exercise": exercises, "Cluster": labels}
    dataframe = pd.DataFrame(data=data)

    return dataframe


def get_CP_factors(direction, max_rank, no_compute=False):
    try:  # Try loading data
        if direction == 'rows':
            factors = np.load("../saved_fig_data/CP_rows_feature_vectors.npy")
        elif direction == 'columns':
            factors = np.load("../saved_fig_data/CP_columns_feature_vectors.npy")
        elif direction == 'tubes':
            factors = np.load("../saved_fig_data/CP_tubes_feature_vectors.npy")
        if factors.shape[1] < max_rank:  # Check if there are insufficient factors, if so, compute the extra factors
            if no_compute:
                return
            print("Computing factors!")
            compute_CP_factors(max_rank)
            return get_CP_factors(direction, max_rank, True)  # no_compute=True to prevent infinite loops
        else:
            return factors[:, 0:max_rank]
    except OSError:  # If this fails, data does not exist. Compute data.
        if no_compute:
            return
        print("Computing factors!")
        compute_CP_factors(max_rank)
        return get_CP_factors(direction, max_rank, True)  # no_compute=True to prevent infinite loops


def compute_CP_factors(max_rank):
    tensor = np.load("../saved_tensors/full_tensor.npy")
    factors = parafac(tensor, rank=max_rank, normalize_factors=False)[1]
    rows_feature_vectors = factors[2]
    columns_feature_vectors = factors[1]
    tubes_feature_vectors = factors[0]
    np.save("../saved_fig_data/CP_rows_feature_vectors", rows_feature_vectors)
    np.save("../saved_fig_data/CP_columns_feature_vectors", columns_feature_vectors)
    np.save("../saved_fig_data/CP_tubes_feature_vectors", tubes_feature_vectors)


def get_overview():
    dal = DAL("../../data/amie-kinect-data.hdf")
    overview = dal.overview()
    df = pd.DataFrame(overview)
    return df


def k_means(vectors, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(vectors)
    return kmeans.cluster_centers_, kmeans.labels_


np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_rows", 1000)

# print(get_overview())

# frame = cluster("vector_aca_t", 5, 'rows', 25, 3)
# print(frame)
# frame = cluster("cp", 5, 'rows', 25, 3)
# print(frame)

# tensor = np.load("saved_tensors/full_tensor.npy")
# cp_facts = get_CP_decomposition(tensor, 10)[1]
# feature_vectors = cp_facts[2]
# print(feature_vectors.shape)
# print(feature_vectors)
