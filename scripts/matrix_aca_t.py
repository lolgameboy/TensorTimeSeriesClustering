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
        matrix_residu = abs(tensor[k,:,:] - decomp.matrix_at(k))

        # find biggest element in this slice (this is delta)
        (i, j) = argmax_matrix(matrix_residu)

        # calculate residu of tube of delta
        tube_residu = abs(tensor[:,i,j] - decomp.tube_at(i, j))

        # find biggest element in tube (don't pick delta again)
        k = argmax_vector(tube_residu, k) #TODO prevent picking delta again

        # add term
        decomp.add(tensor[k,i,j], matrix_residu, tube_residu)

    # return decomposition
    return decomp

test_tensor = build_tensor(10, 5)
decomp = matrix_aca_t(test_tensor, 3)

print(test_tensor)
print(decomp.full_tensor)