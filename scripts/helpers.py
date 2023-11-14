import random


def sample_max_element(matrix, tries):
    """Takes n elements from the given numpy matrix and returns the position of the largest one"""
    max_e = 0
    max_pos = [0, 0]

    for k in range(tries):
        i = random.randrange(0, len(matrix))
        j = random.randrange(0, len(matrix[0]))
        e = matrix[i, j]  # Normally, this is where you'd perform an expensive DTW operation
        if max_e < e:
            max_e = e
            max_pos = [i, j]

    return max_e, max_pos