import numpy as np
from aca import *
import tensorly as tl

def test_aca():
    passed = 0
    failed = 0

    def test_dim_rank(dim, rank, passed, failed):
        matrix = np.random.randn(dim, dim)
        try:
            decomp, _ = aca(matrix, rank)
            norm = np.linalg.norm(matrix - decomp.full_matrix()) / np.linalg.norm(matrix)
            print(f'rank {rank} | aca norm: {norm}')

            factors = tl.decomposition.parafac(matrix, rank=rank)
            cp = tl.cp_to_tensor(factors)
            normcp = np.linalg.norm(matrix - cp) / np.linalg.norm(matrix)
            print(f'rank {rank} | cp  norm: {normcp}')

            return passed + 1, failed, norm
        except Exception as e:
            print(f'dimension {dim}, rank {rank} got error {e}')
            return passed, failed + 1, -1

        return passed, failed, norm

    print("Dim 10x10")
    # test 10x10
    for rank in [1, 5, 10]:
        passed, failed, norm = test_dim_rank(10, rank, passed, failed)
        
        if 0 <= norm <= 1:
            passed += 1
        else:
            failed += 1

    print("\nDim 100x100")
    # test 100x100
    for rank in [1, 10, 50, 75, 100]:
        passed, failed, norm = test_dim_rank(100, rank, passed, failed)
        
        if 0 <= norm <= 1:
            passed += 1
        else:
            failed += 1
    
    print("\nDim 500x500")
    # test 500x500
    for rank in [1, 10, 50, 100, 150]:
        passed, failed, norm = test_dim_rank(500, rank, passed, failed)
        
        if 0 <= norm <= 1:
            passed += 1
        else:
            failed += 1

    return 100*passed/(passed + failed)

     

def test_matrix_aca_t():
    pass

def test_vector_aca_t():
    pass

print(f'{test_aca()}% of tests passed')
