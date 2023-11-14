import numpy as np

# returns adaptable vector ACA-T decomposition of given tensor
#  > tensor: 3d numpyarray of full tensor to decompose
#  > max_rank: amount of terms to include in decomposition
#  > max_approx: amount of terms to approximate matrix in a single term with ACA
def vector_aca_t(tensor, max_rank, max_approx):
    # initialise decomposition
    decomp = Tensordecomp()

    # sample some elements of tensor (10 different elements)
    k = sample_tensor(tensor, 10)

    # loop until at max_rank
    for rank in range(max_rank):
        #TODO
        pass

    # return decomposition
    return decomp