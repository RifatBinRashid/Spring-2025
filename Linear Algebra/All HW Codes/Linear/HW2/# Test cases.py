import numpy as np

def create_matrix_B(N):
    B = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            B[i, j] = 1 / (i + j + 3)  # Adjusted for 0-based indexing
    return B

def compute_determinants(N_values):
    for N in N_values:
        B = create_matrix_B(N)
        determinant = np.linalg.det(B)
        print(f"Determinant for N = {N}: {determinant}")
        if N == 10:
            print(f"Matrix for N = 10:\n{B}")

# Values of N to compute
N_values = [3, 4, 5, 10, 20]

# Compute and display determinants
compute_determinants(N_values)