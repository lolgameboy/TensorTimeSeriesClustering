import random

# Takes 'tries' elements from the given numpy matrix and returns the position of the largest one
def sample_matrix(matrix, tries):
    (n, m) = matrix.shape
    max_e = -99999
    max_pos = (-1, -1)

    for k in range(tries):
        i = random.randrange(0, n)
        j = random.randrange(0, m)
        e = matrix[i, j]  # Normally, this is where you'd perform an expensive DTW operation
        if max_e < e:
            max_e = e
            max_pos = (i, j)

    return max_e, max_pos

# Takes 'tries' elements from the given numpy tensor and returns the position of the largest one
def sample_tensor(tensor, tries):
    (K, N, M) = tensor.shape
    max_e = -99999
    max_pos = (-1, -1, -1)

    for k in range(tries):
        k = random.randrange(0, K)
        i = random.randrange(0, N)
        j = random.randrange(0, M)
        e = tensor[k, i, j]  # Normally, this is where you'd perform an expensive DTW operation
        if max_e < e:
            max_e = e
            max_pos = (k, i, j)

    return max_e, max_pos

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

# find biggest element in vector and return its index
def argmax_vector(vector):
    n = vector.size
    max_e = -99999
    max_pos = -1

    for i in range(n):
        e = vector[i]
        if max_e < e:
            max_e = e
            max_pos = i

    return i