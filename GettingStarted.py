import numpy
import numpy as np
import random


def ACA(m, n, rank):
    # Sample 10 random elements, take row and column containing largest element
    matrix = np.full((m, n), -1, float)  # Init empty array
    maxE = 0
    maxPos = [0, 0]

    for i in range(10):
        x = random.randrange(0, n)
        y = random.randrange(0, m)
        e = random.random()  # Normally, this is where you'd perform an expensive DTW operation
        matrix[y, x] = e  # Store distance for later use
        if maxE < e:
            maxE = e
            maxPos = [y, x]

    for i in range(m):
        if matrix[i, maxPos[1]] == -1:
            matrix[i, maxPos[1]] = random.random()
    for i in range(n):
        if matrix[maxPos[0], i] == -1:
            matrix[maxPos[0], i] = random.random()

    # Initial decomp consists of these first 2 vectors
    decomp = np.matrix([matrix[:, maxPos[1]] / maxE, matrix[maxPos[0], :]])

    for i in range((rank - 1) * 2):  # Times 2 because we need 2 vectors per rank
        # find max element of last vector in decomposition
        maxE = 0
        maxIndex = 0
        lastV = decomp[-1]
        for k in range(lastV.shape[1]):
            if maxE < lastV[0, k]:
                maxE = lastV[0, k]
                maxIndex = k

        if i % 2 == 0:  # If i is even, the next vector we need in our decomposition is a column
            for l in range(m):
                if matrix[l, maxPos[1]] == -1:
                    matrix[l, maxPos[1]] = random.random()
            residu = np.matrix(matrix[:, maxPos[1]]) - column_at(decomp, maxPos[1])
            decomp = np.vstack((decomp, residu))


def element_at(decomp, i, j):
    pass


def row_at(decomp, i):
    pass


def column_at(decomp, j):
    pass


ACA(5, 5, 2)
