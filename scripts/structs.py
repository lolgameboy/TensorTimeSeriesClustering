import numpy as np


class MatrixDecompTerm:
    def __init__(self, delta, column, row):
        self.delta = delta
        self.column = np.array(column)
        self.row = np.array(row)

    def element_at(self, i, j):
        """Returns the element at position (i,j) of the matrix this term represents"""
        return self.column[i] * self.row[j] * self.delta


class TensorDecompTerm:
    def __init__(self, delta, column, row, tube):
        self.delta = delta
        self.column = np.array(column)
        self.row = np.array(row)
        self.tube = np.array(tube)

    def element_at(self, i, j, k):
        """Returns the element at position (i,j,k) of the tensor this term represents"""
        return self.column[i] * self.row[j] * self.tube[k] * self.delta

    def full_tensor(self):
        """Returns the full tensor this term represents"""
        a = self.column
        b = self.row
        c = self.tube
        return np.outer(a, np.outer(b, c)).reshape(a.shape[0], b.shape[0], c.shape[0]) * self.delta


class MatrixDecomp:
    def __init__(self, n, m, term_list=[]):
        self.n = n
        self.m = m
        self.term_list = term_list
    
    def add(self, factor, column, row):
        term = MatrixDecompTerm(factor, column, row)
        self.term_list.append(term)

    def element_at(self, i, j):
        e = 0
        tlist = self.term_list
        for k in range(len(tlist)):
            e += tlist[k].element_at(i, j)
        return e

    def row_at(self, i):
        row = []
        for j in range(self.m):
            row.append(self.element_at(i, j))
        return np.array(row)

    def column_at(self, j):
        column = []
        for i in range(self.n):
            column.append(self.element_at(i, j))
        return np.array(column)

    def full_matrix(self):
        tlist = self.term_list
        matrix = np.outer(tlist[0].column, tlist[0].row) * tlist[0].delta
        for k in range(1, len(tlist)):
            matrix += np.outer(tlist[k].column, tlist[k].row) * tlist[k].delta
        return matrix


class TensorDecomp:
    def __init__(self, K, N, M, term_list=[]):
        self.K = K
        self.N = N
        self.M = M
        self.term_list = term_list

    def add(self, factor, column, row, tube):
        term = TensorDecompTerm(factor, column, row, tube)
        self.term_list.append(term)

    def element_at(self, i, j, k):
        e = 0
        tlist = self.term_list
        for t in range(len(tlist)):
            e += tlist[t].element_at(i, j, k)
        return e

    def row_at(self, i, k):
        row = []
        for j in range(self.M):
            row.append(self.element_at(i, j, k))
        return np.array(row)

    def column_at(self, j, k):
        column = []
        for i in range(self.N):
            column.append(self.element_at(i, j, k))
        return np.array(column)

    def tube_at(self, i, j):
        tube = []
        for k in range(self.K):
            tube.append(self.element_at(i, j, k))
        return np.array(tube)

    def matrix_at(self, k):
        matrix = []
        for i in range(self.N):
            matrix.append(self.row_at(i, k))
        return np.array(matrix)

    def full_tensor(self):
        tlist = self.term_list
        tensor = np.outer(tlist[0].column, tlist[0].row, tlist[0].tube) * tlist[0].delta
        for t in range(1, len(tlist)):
            tensor += tlist[t].full_tensor
        return tensor

