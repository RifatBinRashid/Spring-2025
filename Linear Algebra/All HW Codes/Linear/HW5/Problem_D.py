import numpy as np
from tkinter import Tk, filedialog

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

def main():
    # Set up Tkinter file dialog
    root = Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Select image .npy files",
        filetypes=[("NumPy files", "*.npy")]
    )
    
    if not file_paths:
        print("No files selected.")
        return
    
    for file in file_paths:
        print("\n----------------------------------")
        print(f"Processing file: {file}")
        # Load the image matrix from the .npy file
        A = np.load(file)
        print(f"Matrix shape: {A.shape}")
        
        # Compute the SVD factors via the A^T A method
        U, SigmaT_Sigma, U1, Sigma1, V1 = svd_via_ata(A)
        
        # Print the results
        np.set_printoptions(precision=4, suppress=True)
        print("\nU1 (columns corresponding to positive eigenvalues):")
        print(U1)
        print("\nΣ1 (diagonal matrix of singular values):")
        print(Sigma1)
        print("\nV1 (computed as A U1 Σ1^{-1}):")
        print(V1)
        
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
