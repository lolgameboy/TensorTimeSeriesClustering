import random
import numpy as np




def sample_matrix(matrix, amount):
    '''
    Takes 'amount' elements at random from the given numpy matrix and returns list with elements ((i,j), value)
    :param matrix: Matrix to sample
    :param amount: Amount of samples
    :return: A list of samples and their position in given matrix
    '''
    N, M = matrix.shape
    samples = []

    for _ in range(amount):
        i = random.randrange(0, N)
        j = random.randrange(0, M)
        
        samples.append(((i,j), matrix[i,j]))

    return samples


def sample_tensor(tensor, amount):
    '''
    Takes elements at random from a numpy tensor and returns list with elements ((k,i,j), value)
    :param tensor: Numpy tensor to sample
    :param amount: Amount of samples to return
    :return: List of samples and their position in given tensor
    '''

    K, N, M = tensor.shape
    samples = []

    for _ in range(amount):
        k = random.randrange(0, K)
        i = random.randrange(0, N)
        j = random.randrange(0, M)
        
        samples.append(((k,i,j), tensor[k,i,j]))

    return samples


def max_abs_samples(samples):
    '''
    :param samples: List of samples (from sample_matrix or sample_tensor)
    :return: tuple (index, value) of sample with biggest absolute value
    '''
    max_e = 0
    max_pos = None

    for sample in samples:
        pos, e = sample

        if abs(max_e) <= abs(e):
            max_e = e
            max_pos = pos

    return max_pos, max_e


def update_samples_tensor(samples, tensor, decomp):
    for u, sample in enumerate(samples):
        (k, i, j), e = sample
        e = tensor[k,i,j] - decomp.element_at(k, i, j)
        samples[u] = ((k, i, j), e)


def argmax_matrix(matrix):
    '''Find biggest element in matrix and return its index'''
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


def argmax_vector(vector, ignore_index=-1):
    '''Find biggest element in vector (ignoring ignore_index) and return its index'''
    n = vector.size
    max_e = -99999
    max_pos = -1

    for i, e in enumerate(vector):
        if max_e < e and i != ignore_index:
            max_e = e
            max_pos = i

    return max_pos