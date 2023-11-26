import numpy as np
from helpers import *
from structs import *


def aca(test_matrix, max_rank, start_sample=None):
    n, m = test_matrix.shape
    decomp = MatrixDecomp(n, m, [])

    if start_sample is None:
        i, j = argmax_samples(sample_matrix(test_matrix, 5))
    else:
        i, j = start_sample

    for rank in range(max_rank):
        new_column = test_matrix[:, j] - decomp.column_at(j)

        i = argmax_vector(new_column, i)

        new_row = test_matrix[i, :] - decomp.row_at(i)
        new_factor = 1 / (test_matrix[i, j] - decomp.element_at(i, j))

        j = argmax_vector(new_row, j)

        decomp.add(new_factor, new_column, new_row)
    
    return decomp, (i, j)