import numpy as np
from helpers import *
from structs import *
from tensor import build_tensor


# returns matrix ACA-T decomposition of given tensor
#  > tensor: 3d numpyarray of full tensor to decompose
#  > max_rank: amount of terms to include in decomposition
def matrix_aca_t(tensor, max_rank):
    # initialise decomposition
    K, N, M = tensor.shape
    decomp = TensorDecomp(K, N, M, [])

    # sample some elements of tensor
    (k, _, _) = argmax_samples(sample_tensor(tensor, 3))

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

test_tensor = np.load("saved_tensors/full_tensor.npy")[0:15,0:20,0:20]
decomp = matrix_aca_t(test_tensor, 5)

np.set_printoptions(suppress=True, precision=3)

for i in range(16):
    decomp = matrix_aca_t(test_tensor, i)
    
    print(f'rank {i} with norm = {np.linalg.norm(test_tensor-decomp.full_tensor())/np.linalg.norm(test_tensor)}')