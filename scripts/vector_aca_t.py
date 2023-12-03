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
    decomp = TensorDecomp(K, N, M, [])

    # sample some elements of tensor
    #S = sample_tensor(tensor, 3)
    S = [((0,0,2), tensor[0,0,2]), ((2,0,0), tensor[2,0,2]), ((2,1,0), tensor[2,1,0])]
    (k, i, j) = argmax_samples(S)

    for sample in S:
        idx, v = sample
        print(idx)

    for rank in range(max_rank):

        # calculate ACA decomp of slice
        matrix_residu = tensor[k,:,:] - decomp.matrix_at(k)
        aca_decomp, (i, j) = aca(matrix_residu, max_approx, (i, j))

        # calculate residu of tube of delta
        tube_residu = tensor[:,i,j] - decomp.tube_at(i, j)

        # add term
        decomp.add(1/tube_residu[k], tube_residu, aca_decomp)

        # find biggest element in tube (don't pick delta again)
        k = argmax_vector(abs(tube_residu), k)

        # update samples to pick new (k, i, j)
        update_samples_tensor(S, aca_decomp, tube_residu, 1/tube_residu[k])
        (k, i, j) = argmax_samples(S)

    # return decomposition
    return decomp

test_tensor = np.load("saved_tensors/full_tensor.npy")[0:3, 0:4, 0:4]

np.set_printoptions(suppress=True, precision=3)

for i in range(1, 3):
    decomp = vector_aca_t(test_tensor, i, 1)

    print(f'rank {i} with norm = {np.linalg.norm(test_tensor - decomp.full_tensor()) / np.linalg.norm(test_tensor)}')