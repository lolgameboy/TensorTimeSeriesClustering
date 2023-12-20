import numpy as np
from helpers import *
from structs import *


def matrix_aca_t(tensor, max_rank):
    '''
    :param tensor: 3d numpyarray of full tensor to decompose
    :param max_rank: amount of terms to include in decomposition
    :return: matrix ACA-T decomposition of given tensor
    '''

    # initialise decomposition
    K, N, M = tensor.shape
    decomp = TensorDecomp(K, N, M, [])

    # sample some elements of tensor
    (k, _, _), _ = max_abs_samples(sample_tensor(tensor, 3))

    for rank in range(max_rank):

        # calculate residu of slice of k
        matrix_residu = tensor[k,:,:] - decomp.matrix_at(k)

        # find biggest element in this slice (this is delta)
        (i, j) = argmax_matrix(abs(matrix_residu))

        # calculate residu of tube of delta
        tube_residu = tensor[:,i,j] - decomp.tube_at(i, j)

        # add term
        decomp.add_matrix_term(1/tube_residu[k], tube_residu, matrix_residu)

        # find biggest element in tube (don't pick delta again)
        k = argmax_vector(abs(tube_residu), k)

    # return decomposition
    return decomp
