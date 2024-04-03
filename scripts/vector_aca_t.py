import numpy as np
from helpers import *
from structs import *

def vector_aca_t(tensor, max_rank, max_approx, count=False):
    '''
    Calculates the decomposition of Vector ACA-T type k (where k := max_approx)
    :param tensor: 3d numpyarray of full tensor to decompose
    :param max_rank: amount of terms to include in decomposition
    :param max_approx: amount of terms to approximate matrix in a single term with ACA
    :param count: should the amount of DTW operation be counted?
    :return: adaptable vector ACA-T decomposition of given tensor and the amount of DTW operations (theoretically) performed if count is true.
    '''

    # initialise decomposition
    K, N, M = tensor.shape
    decomp = TensorDecomp(K, N, M, max_rank)
    dtws = 0

    # sample some elements of tensor
    sample_amount = 10

    S = sample_tensor(tensor, sample_amount)
    (k, i, j), _ = max_abs_samples(S)
    dtws += sample_amount

    for rank in range(max_rank):

        # calculate ACA decomp of slice
        matrix_residu = tensor[k,:,:] - decomp.matrix_at(k)
        aca_decomp, (i, j), dtws_aca = aca(matrix_residu, max_approx, (i, j))
        dtws += dtws_aca

        # calculate residu of tube of delta
        tube_residu = tensor[:,i,j] - decomp.tube_at(i, j)
        dtws += K

        if abs(tube_residu[k]) < 0.000001:
            raise Exception("zero as delta")

        # add term
        decomp.add(1/tube_residu[k], tube_residu, aca_decomp)

        # find biggest element in tube (don't pick delta again)
        k = argmax_vector(abs(tube_residu), k)

        # update samples
        update_samples_tensor(S, tensor, decomp)
        p, v = max_abs_samples(S)
        dtws += sample_amount

        # if max sample is bigger -> choose sample instead
        if v > decomp.element_at(k, i, j):
            (k, i, j) = p

    # return decomposition
    if count:
        return decomp, dtws
    else:
        return decomp


def aca(matrix, max_rank, start_sample=None):
    '''
    An adapted ACA-algorithm specifically for vector_aca_t().

    :return: (a, b, c) with a = the matrix decomposition, 
             b = the position of the greatest element (in abs value) of all used (= theoretically calculated) elements in the given matrix (NOT in the decomposition), 
             c = the amount of DTW operations the algorithm (theoretically) performed.
    '''

    n, m = matrix.shape
    decomp = MatrixDecomp(n, m, max_rank)
    dtws = 0

    if start_sample is None:
        sample_amount = 10
        (i, j), _ = max_abs_samples(sample_matrix(matrix, sample_amount))
        dtws += sample_amount
    else:
        i, j = start_sample
    
    I, J = (i, j)

    for rank in range(max_rank):
        column_residu = matrix[:, j] - decomp.column_at(j)
        dtws += n

        i = argmax_vector(abs(column_residu))

        row_residu = matrix[i, :] - decomp.row_at(i)
        dtws += m

        factor = 1 / (row_residu[j])
        decomp.add(factor, column_residu, row_residu)

        j = argmax_vector(abs(row_residu), j)

        # row i and column j used so we update the max elem
        j2 = argmax_vector(abs(matrix[i, :]))
        i2 = argmax_vector(abs(matrix[:, j]))

        if abs(matrix[i, j2]) > abs(matrix[I, J]):
            I, J = i, j2
        if abs(matrix[i2, j]) > abs(matrix[I, J]):
            I, J = i2, j

    return decomp, (I, J), dtws