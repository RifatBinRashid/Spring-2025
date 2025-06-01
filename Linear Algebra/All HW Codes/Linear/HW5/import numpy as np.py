import numpy as np
from scipy.linalg import svd

def compute_svd_components(A):
    # Compute A^T A
    ATA = np.dot(A.T, A)
    
    # Spectral decomposition of A^T A (eigenvalues and eigenvectors)
    eigenvalues, U = np.linalg.eigh(ATA)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    U = U[:, idx]
    
    # Compute Sigma^T Sigma (diagonal matrix of eigenvalues)
    SigmaTSigma = np.diag(eigenvalues)
    
    # Extract positive singular values and corresponding U_1
    positive_mask = eigenvalues > 1e-10  # Threshold for positive values
    Sigma1 = np.diag(np.sqrt(eigenvalues[positive_mask]))
    U1 = U[:, positive_mask]
    
    # Compute V1
    V1 = np.dot(A, np.dot(U1, np.linalg.inv(Sigma1)))
    
    return U, SigmaTSigma, U1, V1, Sigma1

# Example usage:
# A = np.array(...)  # Load your matrix here
A = np.array([[1, 2], [3, 4], [5, 6]])  # A 3x2 matrix
U, SigmaTSigma, U1, V1, Sigma1 = compute_svd_components(A)
for row in Sigma1:
    print(row)
# Print shapes and some properties
print("Shape of U:", U.shape)               # (2, 2)
print("Shape of SigmaTSigma:", SigmaTSigma.shape)  # (2, 2)
print("Shape of U1:", U1.shape)            # (2, r), where r is the rank of A
print("Shape of V1:", V1.shape)            # (3, r)
print("Shape of Sigma1:", Sigma1.shape)    # (r, r)

# Print singular values (diagonal of Sigma1)
print("Singular values:", np.diag(Sigma1))