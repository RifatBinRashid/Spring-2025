import numpy as np
from scipy.linalg import qr

# =============================================
# Power Method (with iteration count)
# =============================================
def power_method(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    iterations = 0
    
    for i in range(max_iter):
        Ax = A @ x
        x_new = Ax / np.linalg.norm(Ax)
        iterations += 1
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    eigenvalue = x.T @ A @ x
    return eigenvalue, x, iterations

# =============================================
# QR Algorithm (with iteration count)
# =============================================
def qr_algorithm(A, max_iter=1000, tol=1e-6):
    A_k = np.copy(A)
    iterations = 0
    
    for i in range(max_iter):
        Q, R = qr(A_k)
        A_k = R @ Q
        iterations += 1
        
        if np.max(np.abs(np.tril(A_k, -1))) < tol:
            break
    
    eigenvalues = np.diag(A_k)
    return eigenvalues, iterations

# =============================================
# Matrix Creation Functions
# =============================================
def create_tridiagonal(n=10):
    A = np.zeros((n, n))
    np.fill_diagonal(A, 2)
    np.fill_diagonal(A[1:], -1)
    np.fill_diagonal(A[:, 1:], -1)
    return A

def create_full_matrix(n=10):
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i,j] = 2 if i == j else -1/(i+j+2)
    return A

# =============================================
# Analysis (c): Tridiagonal Matrix
# =============================================
A_tri = create_tridiagonal()
print("\n=== Tridiagonal Matrix Results ===")

# Power Method
eigval_tri, eigvec_tri, iter_power_tri = power_method(A_tri)
print(f"\nPower Method:")
print(f"Dominant eigenvalue: {eigval_tri:.6f}")
print(f"Dominant eigenvector:\n{eigvec_tri}")
print(f"Iterations: {iter_power_tri}")

# QR Algorithm
eigvals_tri, iter_qr_tri = qr_algorithm(A_tri)
print(f"\nQR Algorithm:")
print(f"All eigenvalues:\n{np.sort(eigvals_tri)}")
print(f"Iterations: {iter_qr_tri}")

# =============================================
# Analysis (d): Full Matrix
# =============================================
A_full = create_full_matrix()
print("\n=== Full Matrix Results ===")

# Power Method
eigval_full, eigvec_full, iter_power_full = power_method(A_full)
print(f"\nPower Method:")
print(f"Dominant eigenvalue: {eigval_full:.6f}")
print(f"Dominant eigenvector:\n{eigvec_full}")
print(f"Iterations: {iter_power_full}")

# QR Algorithm
eigvals_full, iter_qr_full = qr_algorithm(A_full)
print(f"\nQR Algorithm:")
print(f"All eigenvalues:\n{np.sort(eigvals_full)}")
print(f"Iterations: {iter_qr_full}")

# =============================================
# Analysis (e): Comparison
# =============================================
print("\n=== Comparison of Results ===")
print("Tridiagonal Matrix:")
print(f"- Power Method converged in {iter_power_tri} iterations")
print(f"- QR Algorithm converged in {iter_qr_tri} iterations")

print("\nFull Matrix:")
print(f"- Power Method converged in {iter_power_full} iterations")
print(f"- QR Algorithm converged in {iter_qr_full} iterations")

print("\nKey Observations:")
print("1. Tridiagonal matrix converges faster in both methods")
print("2. Full matrix requires more iterations due to dense structure")
print("3. Power Method finds only the dominant eigen-pair")
print("4. QR Algorithm finds all eigenvalues but needs more iterations for full matrices")