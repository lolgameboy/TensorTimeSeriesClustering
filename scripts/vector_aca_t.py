import numpy as np

# returns adaptable vector ACA-T decomposition of given tensor
#  > tensor: 3d numpyarray of full tensor to decompose
#  > max_rank: amount of terms to include in decomposition
#  > max_approx: amount of terms to approximate matrix in a single term with ACA
def vector_aca_t(tensor, max_rank, max_approx):
    # initialise decomposition
    (K, N, M) = tensor.shape
    decomp = Tensordecomp()

    # sample some elements of tensor (10 different elements)
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
        decomp.add(tensor[k,i,j], aca_decomp, tube_residu) #TODO is dit correct?

    # return decomposition
    return decomp