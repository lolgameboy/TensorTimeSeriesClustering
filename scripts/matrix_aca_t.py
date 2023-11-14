import numpy as np


# returns matrix ACA-T decomposition of given tensor
#  > tensor: 3d numpyarray of full tensor to decompose
#  > max_rank: amount of terms to include in decomposition
def matrix_aca_t(tensor, max_rank):
    # initialise decomposition
    decomp = Tensordecomp()

    # sample some elements of tensor (10 different elements)
    k = sample_tensor(tensor, 10)

    # loop until at max_rank
    for rank in range(max_rank):

        # calculate residu of full slice of Index
        matrix = tensor[k,:,:]
        decomp_matrix = decomp.matrix_at(k)
        matrix_residu = abs(matrix - decomp_matrix)

        # find biggest element in this slice (this is delta)
        (i, j) = argmax_matrix(residu)

        # calculate residu of full tube delta
        tube = tensor[:,j,i]
        decomp_tube = decomp.tube_at(i, j)
        tube_residu = abs(tube - decomp_tube)

        # find biggest element in tube and make k the slice index of this element
        k = argmax_tube(residu)

        # add term (delta, slice_residu, tube_residu) in decomposition
        decomp.add(tensor[k,i,j], matrix_residu, tube_residu)

    # return decomposition
    return decomp

def sample_tensor(tensor, tries):
    max_e = 0
    max_pos = [0, 0, 0]

    for k in range(tries):
        k = random.randrange(0, len(tensor))
        j = random.randrange(0, len(tensor[0]))
        i = random.randrange(0, len(tensor[0][0]))
        e = tensor[k, j, i]  # Normally, this is where you'd perform an expensive DTW operation
        if max_e < e:
            max_e = e
            max_pos = [i, j, k]

    return max_e, max_pos