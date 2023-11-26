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
    S = sample_tensor(tensor, 10)
    (k, i, j) = argmax_samples(S)

    for rank in range(max_rank):

        # calculate ACA decomp of slice
        matrix_residu = tensor[k,:,:] - decomp.matrix_at(k) #TODO how to make aca work without 'calculating' full matrix slice to give as parameter?
        aca_decomp, (i, j) = aca(matrix_residu, max_approx, (i, j))  #TODO let aca also return index of next delta

        # calculate residu of tube of delta
        tube_residu = tensor[:,i,j] - decomp.tube_at(i, j) # NOTE AAN LOWIE: Ik denk da ge hierbij nog rekening
                                                           # moet houden met de matrix_decomp die ge net hebt berekend
                                                           # omda ge nu op positie tube_residu[k] een andere value kunt
                                                           # hebben dan bij aca_decomp.element_at(i,j)

        # add term
        decomp.add(1/tube_residu[k], tube_residu, aca_decomp)

        # find biggest element in tube (don't pick delta again)
        k = argmax_vector(abs(tube_residu), k)

        # update samples to pick new (k, i, j)
        # TODO: update_samples_tensor(S, aca_decomp, tube_residu, tensor[k,i,j])
        # TODO: (k, i, j) = argmax_samples(S)

    # return decomposition
    return decomp

test_tensor = np.load("../saved_tensors/full_tensor.npy")[0:15, 0:20, 0:20]

np.set_printoptions(suppress=True, precision=3)

for i in range(16):
    decomp = vector_aca_t(test_tensor, i, 3)

    print(f'rank {i} with norm = {np.linalg.norm(test_tensor - decomp.full_tensor()) / np.linalg.norm(test_tensor)}')