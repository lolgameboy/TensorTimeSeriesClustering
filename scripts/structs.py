import numpy as np


class TensorDecompTerm:
    def __init__(self, delta, tube, matrix_decomp):
        self.delta = delta
        self.tube = tube
        self.matrix_decomp = matrix_decomp

    def element_at(self, k, i, j):
        """Returns the element at position (k,i,j) of the tensor this term represents"""
        return self.matrix_decomp.element_at(i, j) * self.tube[k] * self.delta

    def full_tensor(self):
        """Returns the full tensor this term represents"""
        a = self.matrix_decomp.full_matrix()
        return self.delta * np.outer(self.tube, a).reshape(self.tube.size, a.shape[0], a.shape[1])


class TensorDecompTermMatrix:
    def __init__(self, delta, tube, matrix):
        self.delta = delta
        self.tube = tube
        self.matrix = matrix

    def element_at(self, k, i, j):
        return self.matrix[i, j] * self.tube[k] * self.delta

    def full_tensor(self):
        return self.delta * np.outer(self.tube, self.matrix).reshape(self.tube.size, self.matrix.shape[0], self.matrix.shape[1])


class MatrixDecomp:
    def __init__(self, n, m, max_rank):
        self.n = n
        self.m = m
        self.max_rank = max_rank
        self.factors = np.zeros(max_rank)
        self.rows = np.zeros((max_rank, m))
        self.columns = np.zeros((max_rank, n))
        self.rank = 0

    def add(self, factor, column, row):
        self.factors[self.rank] = factor
        self.columns[self.rank, :] = column
        self.rows[self.rank, :] = row
        self.rank += 1

    def element_at(self, i, j):
        e = 0
        for k in range(self.rank):
            e += self.rows[k, j] * self.columns[k, i] * self.factors[k]
        return e

    def row_at(self, i):
        row = np.zeros(self.m)
        for j in range(self.m):
            row[j] = self.element_at(i, j)
        return row

    def column_at(self, j):
        column = np.zeros(self.n)
        for i in range(self.n):
            column[i] = self.element_at(i, j)
        return np.array(column)

    def full_matrix(self):
        matrix = np.transpose(self.columns) * np.diag(self.factors) * self.rows
        return matrix


class TensorDecomp:
    def __init__(self, K, N, M, max_rank):
        self.K = K
        self.N = N
        self.M = M
        self.term_list = []

    def add(self, delta, tube, matrix_decomp):
        term = TensorDecompTerm(delta, tube, matrix_decomp)
        self.term_list.append(term)

    def add_matrix_term(self, delta, tube, matrix):
        term = TensorDecompTermMatrix(delta, tube, matrix)
        self.term_list.append(term)

    def element_at(self, k, i, j):
        e = 0
        terms = self.term_list
        for t in range(len(terms)):
            e += terms[t].element_at(k, i, j)
        return e

    def tube_at(self, i, j):
        tube = np.zeros(self.K)
        for k in range(self.K):
            tube[k] = self.element_at(k, i, j)
        return tube

    def row_at(self, k, i):
        row = np.zeros(self.M)
        for j in range(self.M):
            row[j] = self.element_at(k, i, j)
        return row

    def column_at(self, k, j):
        column = np.zeros(self.N)
        for i in range(self.N):
            column[i] = self.element_at(k, i, j)
        return column

    def matrix_at(self, k):
        matrix = np.zeros((self.N, self.M))
        for i in range(self.N):
            matrix[i] = self.row_at(k, i)
        return matrix

    def full_tensor(self):
        terms = self.term_list
        tensor = np.zeros((self.K, self.N, self.M))
        for t in range(len(terms)):
            tensor += terms[t].full_tensor()
        return tensor
