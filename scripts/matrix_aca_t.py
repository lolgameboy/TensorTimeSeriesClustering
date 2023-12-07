import numpy as np
from helpers import *
from structs import *
from tensor import build_tensor
import matplotlib.pyplot as plt
import statistics as stat


def matrix_aca_t(tensor, max_rank):
    '''
    :param tensor: 3d numpyarray of full tensor to decompose
    :param max_rank: amount of terms to include in decomposition
    :return: matrix ACA-T decomposition of given tensor
    '''

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


# Plot relative error for different ranks of decomposition.
tensor = np.load("saved_tensors/full_tensor.npy")
t_norm = np.linalg.norm(tensor)

ranks = range(5, 51, 5)
x_pos = len(ranks)
avgs = []
stdevs = []

for rank in ranks:
    rel_errs = []
    for i in range(50):
        dp = matrix_aca_t(tensor, rank)
        rel_errs.append(np.linalg.norm(tensor-dp.full_tensor())/t_norm)

    avgs.append(stat.mean(rel_errs))
    stdevs.append(stat.stdev(rel_errs))

    print(f'rank {rank} with norm = {avgs[-1]} | stdev = {stdevs[-1]}')

 
plt.subplots()
plt.bar(ranks, avgs, width=2, yerr=stdevs, color='orange')

plt.xlabel('Rang')
plt.ylabel('Relatieve fout')
plt.title('Relatieve fout van matrix ACA-T per rang')
plt.xticks(ranks)

plt.savefig('figures/rel_fout_matrix_aca_t.png')
plt.show()
