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
    def __init__(self, n, m, term_list):
        self.n = n
        self.m = m
        self.term_list = term_list
    
    def add(self, factor, column, row):
        term = MatrixDecompTerm(factor, column, row)
        self.term_list.append(term)

    def element_at(self, i, j):
        e = 0
        terms = self.term_list
        for k in range(len(terms)):
            e += terms[k].element_at(i, j)
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
        terms = self.term_list
        matrix = np.outer(terms[0].column, terms[0].row) * terms[0].delta
        for k in range(1, len(terms)):
            matrix += np.outer(terms[k].column, terms[k].row) * terms[k].delta
        return matrix


class TensorDecomp:
    def __init__(self, K, N, M, term_list):
        self.K = K
        self.N = N
        self.M = M
        self.term_list = term_list

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
        tube = []
        for k in range(self.K):
            tube.append(self.element_at(k, i, j))
        return np.array(tube)

    def row_at(self, k, i):
        row = []
        for j in range(self.M):
            row.append(self.element_at(k, i, j))
        return np.array(row)

    def column_at(self, k, j):
        column = []
        for i in range(self.N):
            column.append(self.element_at(k, i, j))
        return np.array(column)

    def matrix_at(self, k):
        matrix = []
        for i in range(self.N):
            matrix.append(self.row_at(k, i))
        return np.array(matrix)

    def full_tensor(self):
        terms = self.term_list
        tensor = np.zeros((self.K, self.N, self.M))
        for t in range(len(terms)):
            tensor += terms[t].full_tensor()
        return tensor

