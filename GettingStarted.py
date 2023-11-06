import numpy as np
import random


class DecompTerm:
    def __init__(self, factor, column, row):
        self.factor = factor
        self.column = column
        self.row = row


class Decomp:
    def __init__(self, n, m, term_list):
        self.n = n
        self.m = m
        self.term_list = term_list


def ACA(n, m, rank):
    test_matrix = np.random.random((n, m))

    elem, max_pos = sample_max_element(test_matrix, 5)
    # Initial decomp consists of this first vector
    firstTerm = DecompTerm(1/elem, test_matrix[:, max_pos[1]], test_matrix[max_pos[0], :])

    decomp = Decomp(5, 5, [firstTerm])



    for i in range(0, rank):
        last_row = decomp.term_list[-1].row

        # find max element of last row in decomposition
        max_e = 0
        max_column_index = 0
        for k in range(len(last_row)):
            if max_e < last_row[k]:
                max_e = last_row[k]
                max_column_index = k

        residu = test_matrix - full_matrix(decomp)
        new_column = residu[:, max_column_index]

        max_e = 0
        max_row_index = 0
        for k in range(len(new_column)):
            if max_e < new_column[k]:
                max_e = new_column[k]
                max_row_index = k

        new_row = residu[max_row_index, :]
        new_factor = 1/residu[max_row_index, max_column_index]

        new_term = DecompTerm(new_factor, new_column, new_row)
        decomp.term_list.append(new_term)

    return decomp, test_matrix


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


def element_at(decomp, i, j):
    e = 0
    for k in range(len(decomp.term_list)):
        e += np.outer(decomp.term_list[k].column[i], decomp.term_list[k].row[j]) * decomp.term_list[k].factor
    return e


def row_at(decomp, i):
    row = []
    for j in range(decomp.m):
        row.append(element_at(decomp, i, j))
    return row


def column_at(decomp, j):
    column = []
    for i in range(decomp.n):
        column.append(element_at(decomp, i, j))
    return column


def full_matrix(decomp):
    matrix = np.outer(decomp.term_list[0].column, decomp.term_list[0].row)
    matrix *= decomp.term_list[0].factor
    for k in range(1, len(decomp.term_list)):
        matrix += np.outer(decomp.term_list[k].column, decomp.term_list[k].row) * decomp.term_list[k].factor
    return matrix

# Disable scientific notation in output
np.set_printoptions(suppress=True, precision=3)
dec, mat = ACA(8, 8, 3)
print("RESULTS")
print(mat)
print(full_matrix(dec))
print(mat-full_matrix(dec))
