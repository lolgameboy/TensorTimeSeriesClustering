import numpy as np
from helpers import *
from structs import *


def vector_aca_t(tensor, max_rank, max_approx):
    '''
    :param tensor: 3d numpyarray of full tensor to decompose
    :param max_rank: amount of terms to include in decomposition
    :param max_approx: amount of terms to approximate matrix in a single term with ACA
    :return: adaptable vector ACA-T decomposition of given tensor
    '''

    # initialise decomposition
    K, N, M = tensor.shape
    decomp = TensorDecomp(K, N, M, [])

    # sample some elements of tensor
    S = sample_tensor(tensor, 10)
    (k, i, j) = argmax_samples(S)

    for rank in range(max_rank):
        
        # calculate ACA decomp of slice
        matrix = tensor[k,:,:] #TODO how to make aca work without 'calculating' full matrix slice to give as parameter?
        aca_decomp, (i, j) = aca(matrix, max_approx, (i, j)) #TODO don't sample, just start with (i, j)
                                                             #TODO let aca also return index of next delta
        matrix_residu = abs(matrix - aca_decomp.full_matrix())

        # calculate residu of tube of delta
        tube_residu = abs(tensor[:,i,j] - decomp.tube_at(i, j))

        # find biggest element in tube (don't pick delta again)
        k = argmax_vector(tube_residu, k) #TODO prevent picking delta again

        # add term
        decomp.add_matrix_term(tensor[k,i,j], aca_decomp, tube_residu)

        # update samples to pick new (k, i, j)
        update_samples(S, aca_decomp, tube_residu)
        (k, i, j) = argmax_samples(S)

    # return decomposition
    return decomp