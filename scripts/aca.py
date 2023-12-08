import numpy as np
from helpers import *
from structs import *


def aca(matrix, max_rank, start_sample=None):
    n, m = matrix.shape
    decomp = MatrixDecomp(n, m, [])

    if start_sample is None:
        i, j = argmax_samples(sample_matrix(matrix, 10))
    else:
        i, j = start_sample

    for rank in range(max_rank):
        column_residu = matrix[:, j] - decomp.column_at(j)

        i = argmax_vector(abs(column_residu))

        row_residu = matrix[i, :] - decomp.row_at(i)

        factor = 1 / (row_residu[j])
        decomp.add(factor, column_residu, row_residu)

        j = argmax_vector(abs(row_residu), j)
    
    return decomp, (i, j)

'''
test_tensor = np.load("../saved_tensors/full_tensor.npy")[0:1, 0:5, 0:5]
test_matrix = np.reshape(test_tensor, (5, 5))

test_matrix2 = np.random.random((5,5))

np.set_printoptions(suppress=True, precision=3)

print(test_matrix)
print(test_matrix2)

for i in range(1, 6):
    decomp, _ = aca(test_matrix, i)
    decomp2, _ = aca(test_matrix2, i)

    print(f'rank {i} with norm = {np.linalg.norm(test_matrix - decomp.full_matrix()) / np.linalg.norm(test_matrix)}')
    print(test_matrix - decomp.full_matrix())
    print(f'rank {i} with norm = {np.linalg.norm(test_matrix2 - decomp2.full_matrix()) / np.linalg.norm(test_matrix2)}')
    print(test_matrix2 - decomp2.full_matrix())
'''