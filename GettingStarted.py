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


def slow_ACA(n, m, rank):
    test_matrix = np.random.random((5, 5))
    print(test_matrix)

    elem, max_pos = sample_max_element(test_matrix, 5)

    # Initial decomp consists of this first vector
    firstTerm = DecompTerm(1/elem, test_matrix[:, max_pos[1]], test_matrix[max_pos[0], :])
    decomp = Decomp(5, 5, [firstTerm])

    last_pos = max_pos[0]

    for i in range(0, rank):
        mat = full_matrix(decomp)
        last_v = row_at(decomp, last_pos)

        # find max element of last vector in decomposition
        max_e = 0
        max_index = 0
        for k in range(len(last_v)):
            if max_e < last_v[k]:
                max_e = last_v[k]
                max_index = k

        residu = test_matrix - mat
        new_column = residu[:, k]
        print("new col")
        print(new_column)

        if i % 2 == 0:  # If i is even, the next vector we need in our decomposition is a column
            for l in range(m):
                if matrix[l, maxPos[1]] == -1:
                    matrix[l, maxPos[1]] = random.random()
            residu = np.matrix(matrix[:, maxPos[1]]) - column_at(decomp, maxPos[1])
            decomp = np.vstack((decomp, residu))
        else:
            for l in range(n):
                if matrix[maxPos[0], l] == -1:
                    matrix[maxPos[0], l] = random.random()
            residu = np.matrix(matrix[maxPos[0], :]) - row_at(decomp, maxPos[0])
            decomp = np.vstack((decomp, residu))


def ACA(n, m, rank):
    test_matrix = np.random.random((5, 5))
    print(test_matrix)

    elem, max_pos = sample_max_element(test_matrix, 5)

    # Initial decomp consists of this first vector
    decomp = np.array([test_matrix[:, max_pos[1]] / elem])
    print(decomp)

    for i in range(1, rank * 2):  # Times 2 because we need 2 vectors per rank

        # find max element of last vector in decomposition
        max_e = 0
        max_index = 0
        lastV = decomp[-1]
        print(lastV)
        for k in range(len(lastV)):
            if max_e < lastV[0, k]:
                max_e = lastV[0, k]
                maxIndex = k


        if i % 2 == 0:  # If i is even, the next vector we need in our decomposition is a column
            for l in range(m):
                if matrix[l, maxPos[1]] == -1:
                    matrix[l, maxPos[1]] = random.random()
            residu = np.matrix(matrix[:, maxPos[1]]) - column_at(decomp, maxPos[1])
            decomp = np.vstack((decomp, residu))
        else:
            for l in range(n):
                if matrix[maxPos[0], l] == -1:
                    matrix[maxPos[0], l] = random.random()
            residu = np.matrix(matrix[maxPos[0], :]) - row_at(decomp, maxPos[0])
            decomp = np.vstack((decomp, residu))



    return decomp, matrix


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
        e += decomp.term_list[k].column[i] * decomp.term_list[k].row[j] * decomp.term_list[k].factor
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
    for k in range(len(decomp.term_list)):
        matrix += np.outer(decomp.term_list[0].column, decomp.term_list[0].row)
        matrix *= decomp.term_list[0].factor
    return matrix


dec, mat = slow_ACA(4, 4, 2)
print(dec)
print(mat)
print(mat[2, 3])
print(element_at(dec, 2, 3))
