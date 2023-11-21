import random
import numpy as np

# Takes 'amount' elements at random from the given numpy matrix and returns list with elements ((i,j), value)
def sample_matrix(matrix, amount):
    N, M = matrix.shape
    samples = []

    for _ in range(amount):
        i = random.randrange(0, N)
        j = random.randrange(0, M)
        
        samples.append(((i,j), matrix[i,j]))

    return samples

# Takes 'tries' elements at random from the given numpy tensor and returns list with elements ((k,i,j), value)
def sample_tensor(tensor, amount):
    K, N, M = tensor.shape
    samples = []

    for _ in range(amount):
        k = random.randrange(0, K)
        i = random.randrange(0, N)
        j = random.randrange(0, M)
        
        samples.append(((k,i,j), tensor[k,i,j]))

    return samples

# Takes list of samples (from sample_matrix or sample_tensor)
# Returns index of sample with biggest value
def argmax_samples(samples):
    max_e = -99999
    max_pos = -1

    for sample in samples:
        pos, e = sample

        if max_e < e:
            max_e = e
            max_pos = pos

    return max_pos

# Takes list of samples (from sample_tensor) and a last decomposition term
# Returns list with samples with updated values
# Updated value = value of sample in residu given the new term is added in the decomposition
def update_samples_tensor(samples, matrix_decomp, tube):
    for u, sample in enumerate(samples):
        (k, i, j), e = sample
        e -= matrix_decomp.element_at(i, j) + tube[k]
        samples[u] = ((k, i, j), e)
    


# find biggest element in matrix and return its index
def argmax_matrix(matrix):
    (n, m) = matrix.shape
    max_e = -99999
    max_pos = (-1, -1)

    for i in range(n):
        for j in range(m):
            e = matrix[i, j]
            if max_e < e:
                max_e = e
                max_pos = (i, j)

    return max_pos


# find biggest element in vector (ignoring ignore_index) and return its index
def argmax_vector(vector, ignore_index=-1):
    n = vector.size
    max_e = -99999
    max_pos = -1

    for i, e in enumerate(vector):
        if max_e < e and i != ignore_index:
            max_e = e
            max_pos = i

    return max_pos
