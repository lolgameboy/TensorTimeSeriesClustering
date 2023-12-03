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