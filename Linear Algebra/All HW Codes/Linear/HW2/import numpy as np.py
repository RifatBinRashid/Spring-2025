import numpy as np

def create_matrix_b(N):
    """Creates the N x N matrix B as described in the problem."""
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            B[i, j] = 1 / (i + j + 3)
    return B

def compute_determinant(matrix):
    """Computes the determinant of a matrix using numpy's built-in function."""
    return np.linalg.det(matrix)

# Test cases
Ns = [3, 4, 5, 10, 20]

for N in Ns:
    print(f"Results for N = {N}:")
    B = create_matrix_b(N)
    
    if N == 10:  # Display the full matrix only for N <= 10
        print("Matrix B:")
        for row in B:
            print([f"{val:.5f}" for val in row])
    
    determinant = compute_determinant(B)
    print(f"Determinant of B: {determinant}")
    print("-" * 30) 

# For N=20, let's also check the condition number to understand potential numerical issues
N = 20
B = create_matrix_b(N)
condition_number = np.linalg.cond(B)
print(f"Condition number of B for N=20: {condition_number}") 

# Optional: You can further investigate the condition number and its implications
# for the accuracy of the determinant computation, especially for larger N. 
# Ill-conditioned matrices (high condition number) can lead to unreliable results.