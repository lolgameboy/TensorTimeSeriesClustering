import numpy as np
from helpers import *
from structs import *
from aca import *

def vector_aca_t(tensor, max_rank, max_approx):
    '''
    :param tensor: 3d numpyarray of full tensor to decompose
    :param max_rank: amount of terms to include in decomposition
    :param max_approx: amount of terms to approximate matrix in a single term with ACA
    :return: adaptable vector ACA-T decomposition of given tensor
    '''

    # initialise decomposition
    K, N, M = tensor.shape
    decomp = TensorDecomp(K, N, M, max_rank)

    # sample some elements of tensor
    S = sample_tensor(tensor, 3)
    (k, i, j), _ = max_abs_samples(S)

    for rank in range(max_rank):

        # calculate ACA decomp of slice
        matrix_residu = tensor[k,:,:] - decomp.matrix_at(k)
        aca_decomp, (i, j) = aca(matrix_residu, max_approx, (i, j))

        # calculate residu of tube of delta
        tube_residu = tensor[:,i,j] - decomp.tube_at(i, j)

        if abs(tube_residu[k]) < 0.000001:
            raise Exception("zero as delta")

        # add term
        decomp.add(1/tube_residu[k], tube_residu, aca_decomp)

        # find biggest element in tube (don't pick delta again)
        k = argmax_vector(abs(tube_residu), k)

        # update samples
        #update_samples_tensor_ineff(S, tensor, decomp)
        #p, v = max_abs_samples(S)

        # if max sample is bigger -> choose sample instead
        #if v > decomp.element_at(k, i, j):
        #    (k, i, j) = p

    # return decomposition
    return decomp
