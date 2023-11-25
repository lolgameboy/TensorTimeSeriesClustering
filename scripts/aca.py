import numpy as np
from helpers import *
from structs import *


def aca(test_matrix, max_rank, start_sample=None):
    n, m = test_matrix.shape
    decomp = MatrixDecomp(n, m, [])

    if start_sample is None:
        i, j = sample_matrix(test_matrix, 5)
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

print(np.array([[1,2],[4,5],[7,8]]))
a = np.outer(np.array([[1,2],[4,5],[7,8]]), np.array([10,1,30])).reshape((3,2,3))

# Disable scientific notation in output
np.set_printoptions(suppress=True, precision=3)
mat = np.random.random((10, 10))
dec, _ = aca(mat, 5)
print("RESULTS")
print(mat - dec.full_matrix())

#dt = TensorDecompTerm(1 / 9, [1, 2, 3], [3, 3, 4], [3, 4, 2])
#print(dt.full_tensor())