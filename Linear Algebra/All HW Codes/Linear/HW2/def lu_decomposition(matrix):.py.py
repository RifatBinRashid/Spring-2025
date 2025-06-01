import numpy as np

matrix = [
    [6, 1.25, 1.25, 0],
    [1.25, 6, 1.25, 1.25],
    [1.25, 1.25, 6, 1.25],
    [0, 1.25, 1.25, 6]
]

N = int(input("Enter the size of the symmetric matrix A (N= 4, 10 or 20): "))
A= []
def create_symmetric_matrix(N):
    # Initialize an N×N matrix with zeros
    A = [[ 0.0 for j in range(N)] for i in range(N)]
    
    for i in range (N):
        for j in range (N):
            if i == j:
                A[i][j]= 6
            elif abs(j-i)== 1:
                A[i][j]= 1.25
            elif abs(j-i)== 2:
                A[i][j]= 1.25

    return A

# Generate A for different N
A = create_symmetric_matrix(N)


det = np.linalg.det(A)
print("Determinant (using NumPy) =", det)