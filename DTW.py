import numpy as np


def dynamic_time_warping(series1, series2):
    n = series1.size

    # initialize distance matrix
    matrix = np.full((n, n), -1, float)
    
    # (0, 0) init
    matrix[0, 0] = abs(series1[0] - series2[0])
    # column 0 init
    for i in range(1, n):
        matrix[i, 0] = abs(series1[i] - series2[0]) + matrix[i-1, 0]
    # row 0 init
    for j in range(1, n):
        matrix[0, j] = abs(series1[0] - series2[j]) + matrix[0, j-1]

    # compute rest of matrix 
    for i in range(1, n):
        for j in range(1, n):
            matrix[i, j] = abs(series1[i] - series2[j]) + min(matrix[i-1,j-1], matrix[i-1,j], matrix[i,j-1])

    for i in reversed(range(n)):
        print(matrix[i,:])

    # find path with minimum 'weight' from bottom right corner (n-1, n-1) to top left corner (0, 0)
    i = n-1
    j = n-1
    distance = matrix[n-1, n-1]

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
            distance += matrix[0, j]
        elif j == 0:
            i -= 1
            distance += matrix[i, 0]
        else:
            min_elem = min(matrix[i-1,j-1], matrix[i-1,j], matrix[i,j-1])
            
            if min_elem != matrix[i,j-1]:
                i -= 1
            if min_elem != matrix[i-1,j]:
                j -= 1
            distance += min_elem

    distance += matrix[0, 0]

    return distance

series1 = np.array([1,3,4,9,8,2,1,5,7,3])
series2 = np.array([1,6,2,3,0,9,4,3,6,3])
distance = dynamic_time_warping(series1, series2)
print(distance)


