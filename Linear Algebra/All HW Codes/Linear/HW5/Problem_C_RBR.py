import numpy as np

def svd_via_ata(A, tol=1e-10):
    # Form A^T A
    ATA = A.T @ A
    # Compute spectral decomposition of ATA (ATA is symmetric)
    eigenvalues, U = np.linalg.eigh(ATA)
    # Sort eigenvalues (and corresponding eigenvectors) in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]
    
    # Form Σ^TΣ (diagonal matrix with eigenvalues)
    SigmaT_Sigma = np.diag(eigenvalues)
    
    # Determine rank r (number of positive eigenvalues above tolerance)
    r = np.sum(eigenvalues > tol)
    U1 = U[:, :r]
    singular_values = np.sqrt(eigenvalues[:r])
    Sigma1 = np.diag(singular_values)
    
    # Compute V1 using: A U1 = V1 Σ1  =>  V1 = A U1 Σ1^{-1}
    V1 = A @ U1 @ np.linalg.inv(Sigma1)
    
    return U, SigmaT_Sigma, U1, Sigma1, V1

def input_matrix():
    
    m = int(input("Enter the number of rows of matrix A: "))
    n = int(input("Enter the number of columns of matrix A: "))
    
    A = []
    for i in range(m):
        row_input = input(f"Enter row {i+1} (separate elements by spaces): ")
        # Convert the input string into a list of floats. Change float to int if desired.
        row = list(map(float, row_input.split()))
        
        # Check if the number of entries matches the expected number of columns
        while len(row) != n:
            print(f"Row {i+1} must have {n} elements. Please try again.")
            row_input = input(f"Enter row {i+1} (separate elements by spaces): ")
            row = list(map(float, row_input.split()))
        A.append(row)
    
    return np.array(A)


def main():
    # Ask for matrix A row by row.
    print("Enter matrix A row by row.")
    A = input_matrix()
    print("\nMatrix A:")
    print(A)
    
    # Compute the SVD factors via A^T A method
    U, SigmaT_Sigma, U1, Sigma1, V1 = svd_via_ata(A)
    
    # Set print options for clarity
    np.set_printoptions(precision=4, suppress=True)
    print("\nResults:")
    print("\nU (eigenvectors of A^T A):")
    print(U)
    print("\nΣ^TΣ (diagonal matrix of eigenvalues):")
    print(SigmaT_Sigma)
    print("\nU1 (columns corresponding to positive eigenvalues):")
    print(U1)
    print("\nΣ1 (diagonal matrix of singular values):")
    print(Sigma1)
    print("\nV1 (computed as A U1 Σ1^{-1}):")
    print(V1)

if __name__ == "__main__":
    main()
