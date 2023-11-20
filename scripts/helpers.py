import random

# Takes 'amount' elements at random from the given numpy matrix and returns list with elements ((i,j), value)
def sample_matrix(matrix, amount):
    N, M = tensor.shape
    samples = []

    for _ in range(amount):
        i = random.randrange(0, N)
        j = random.randrange(0, M)
        
        samples.append((i,j), matrix[i,j])

    return np.array(samples)

# Takes 'tries' elements at random from the given numpy tensor and returns list with elements ((k,i,j), value)
def sample_tensor(tensor, amount):
    K, N, M = tensor.shape
    samples = []

    for _ in range(amount):
        k = random.randrange(0, K)
        i = random.randrange(0, N)
        j = random.randrange(0, M)
        
        samples.append((k,i,j), tensor[k,i,j])

    return np.array(samples)

# Returns index of sample with biggest value (both for matrix or tensor)
def argmax_samples(samples):
    max_e = -99999
    max_pos = pos

    for sample in samples:
        pos, e = sample

        if max_e < e:
            max_e = e
            max_pos = pos

    return max_pos

def update_samples(samples, matrix_decomp, tube):
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
def argmax_vector(vector, ignore_index):
    n = vector.size
    max_e = -99999
    max_pos = -1

    for i in range(n):
        e = vector[i]
        if max_e < e and i != ignore_index:
            max_e = e
            max_pos = i

    return max_pos
