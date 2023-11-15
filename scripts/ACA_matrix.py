import numpy as np
from helpers import *
from structs import *


def ACA(n, m, rank):
    test_matrix = np.random.random((n, m))

    elem, (i, j) = sample_matrix(test_matrix, 5)
    # Initial decomp consists of this first vector
    firstTerm = MatrixDecompTerm(1 / elem, test_matrix[:, j], test_matrix[i, :])

    decomp = MatrixDecomp(5, 5, [firstTerm])

    for i in range(0, rank):
        last_row = decomp.term_list[-1].row

        # find max element of last row in decomposition
        max_e = 0
        max_column_index = 0
        for k in range(len(last_row)):
            if max_e < last_row[k]:
                max_e = last_row[k]
                max_column_index = k

        residu = test_matrix - decomp.full_matrix()
        new_column = residu[:, max_column_index]

        max_e = 0
        max_row_index = 0
        for k in range(len(new_column)):
            if max_e < new_column[k]:
                max_e = new_column[k]
                max_row_index = k

        new_row = residu[max_row_index, :]
        new_factor = 1/residu[max_row_index, max_column_index]

        decomp.add(new_factor, new_column, new_row)

    return decomp, test_matrix


# Disable scientific notation in output
np.set_printoptions(suppress=True, precision=3)
dec, mat = ACA(8, 8, 3)
print("RESULTS")
print(mat)
print(dec.full_matrix())
print(mat-dec.full_matrix())

dt = TensorDecompTerm(1/9, [1,2,3], [3,3,4], [3,4,2])
print(dt.full_tensor())
