import numpy as np
from helpers import *
from structs import *
from tensor import build_tensor


# returns matrix ACA-T decomposition of given tensor
#  > tensor: 3d numpyarray of full tensor to decompose
#  > max_rank: amount of terms to include in decomposition
def matrix_aca_t(tensor, max_rank):
    # initialise decomposition
    (K, N, M) = tensor.shape
    decomp = TensorDecomp(K, N, M, [])

    # sample some elements of tensor
    _, (k, _, _) = sample_tensor(tensor, 3)

    for rank in range(max_rank):

        # calculate residu of slice of k
        matrix = tensor[k,:,:] # 'expensive' step
        decomp_matrix = decomp.matrix_at(k)
        matrix_residu = abs(matrix - decomp_matrix)

        # find biggest element in this slice (this is delta)
        (i, j) = argmax_matrix(matrix_residu)

        # calculate residu of tube of delta
        tube = tensor[:,i,j] # 'expensive' step
        decomp_tube = decomp.tube_at(i, j)
        tube_residu = abs(tube - decomp_tube)

        # find biggest element in tube
        k = argmax_vector(tube_residu)

        # add term
        decomp.add(tensor[k,i,j], matrix_residu, tube_residu)

    # return decomposition
    return decomp

test_tensor = build_tensor(10, 5)
decomp = matrix_aca_t(test_tensor, 3)

print(test_tensor)
print(decomp.full_tensor)