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

    def get_row_vectors(self):
        return self.matrix_decomp.rows

    def get_column_vectors(self):
        return self.matrix_decomp.columns

    def get_tube_vector(self):
        return self.tube


class TensorDecompTermMatrix:
    def __init__(self, delta, tube, matrix):
        self.delta = delta
        self.tube = tube
        self.matrix = matrix

    def element_at(self, k, i, j):
        return self.matrix[i, j] * self.tube[k] * self.delta

    def full_tensor(self):
        return self.delta * np.outer(self.tube, self.matrix).reshape(self.tube.size, self.matrix.shape[0], self.matrix.shape[1])

    def get_row_vectors(self):
        return self.matrix


class MatrixDecomp:
    def __init__(self, n, m, max_rank):
        self.n = n
        self.m = m
        self.max_rank = max_rank
        self.factors = np.zeros(max_rank)
        self.rows = np.zeros((max_rank, m))
        self.columns = np.zeros((max_rank, n))
        self.matrix = np.zeros((n, m))
        self.rank = 0

    def add(self, factor, column, row):
        self.factors[self.rank] = factor
        self.columns[self.rank, :] = column
        self.rows[self.rank, :] = row
        self.matrix += np.outer(column, row) * factor
        self.rank += 1

    def element_at(self, i, j):
        return self.matrix[i, j]

    def row_at(self, i):
        return self.matrix[i, :]

    def column_at(self, j):
        return self.matrix[:, j]

    def full_matrix(self):
        return self.matrix


class TensorDecomp:
    def __init__(self, K, N, M, max_rank):
        self.K = K
        self.N = N
        self.M = M
        self.term_list = []
        self.tensor = np.zeros((K, N, M))

    def add(self, delta, tube, matrix_decomp):
        term = TensorDecompTerm(delta, tube, matrix_decomp)
        self.term_list.append(term)
        self.tensor += term.full_tensor()

    def add_matrix_term(self, delta, tube, matrix):
        term = TensorDecompTermMatrix(delta, tube, matrix)
        self.term_list.append(term)
        self.tensor += term.full_tensor()

    def element_at(self, k, i, j):
        return self.tensor[k, i, j]

    def tube_at(self, i, j):
        return self.tensor[:, i, j]

    def row_at(self, k, i):
        return self.tensor[k, i, :]

    def column_at(self, k, j):
        return self.tensor[k, :, j]

    def matrix_at(self, k):
        return self.tensor[k, :, :]

    def full_tensor(self):
        return self.tensor

    def get_row_vectors(self):
        row_vectors = self.term_list[0].get_row_vectors()
        for i in range(1, len(self.term_list)):
            row_vectors = np.concatenate((row_vectors, self.term_list[i].get_row_vectors()))
        return row_vectors

    def get_column_vectors(self):
        column_vectors = self.term_list[0].get_column_vectors()
        for i in range(1, len(self.term_list)):
            column_vectors = np.concatenate((column_vectors, self.term_list[i].get_column_vectors()))
        return column_vectors

    def get_tube_vectors(self):
        tube_vector = self.term_list[0].get_tube_vector()
        tube_vectors = tube_vector.reshape((tube_vector.shape[0], 1)).transpose()
        for i in range(1, len(self.term_list)):
            tube_vector = self.term_list[i].get_tube_vector()
            tube_vector = tube_vector.reshape((tube_vector.shape[0], 1)).transpose()
            tube_vectors = np.concatenate((tube_vectors, tube_vector))
        return tube_vectors
