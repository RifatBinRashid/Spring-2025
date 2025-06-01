import numpy as np
import os
from tkinter import Tk, filedialog
from scipy.io import loadmat

def load_matrix(file):
    ext = os.path.splitext(file)[1].lower()
    if ext == '.npy':
        return np.load(file)
    elif ext == '.mat':
        mat_data = loadmat(file)
        for key in mat_data:
            if not key.startswith('__'):
                return mat_data[key]
        raise ValueError("No valid variable found in the .mat file.")
    else:
        raise ValueError("Unsupported file type: " + ext)

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

def save_results(base_name, U1, Sigma1, V1, save_dir):
    # Construct full paths
    path_U1 = os.path.join(save_dir, base_name + "_U1.txt")
    path_Sigma1 = os.path.join(save_dir, base_name + "_Sigma1.txt")
    path_V1 = os.path.join(save_dir, base_name + "_V1.txt")
    # Save matrices to text files
    np.savetxt(path_U1, U1, fmt="%.4f")
    np.savetxt(path_Sigma1, Sigma1, fmt="%.4f")
    np.savetxt(path_V1, V1, fmt="%.4f")
    print(f"Results saved as:\n  {path_U1}\n  {path_Sigma1}\n  {path_V1}")

def main():
    # Set up Tkinter file dialog to select image files (.npy or .mat)
    root = Tk()
    root.withdraw()  # Hide the root window
    file_paths = filedialog.askopenfilenames(
        title="Select image files (.npy or .mat)",
        filetypes=[("NumPy files", "*.npy"), ("MAT files", "*.mat")]
    )
    
    if not file_paths:
        print("No files selected.")
        return
    
    # Ask for folder where to save the results
    save_dir = filedialog.askdirectory(title="Select folder to save output text files")
    if not save_dir:
        print("No folder selected for saving files. Exiting.")
        return
    
    for file in file_paths:
        print("\n----------------------------------")
        print(f"Processing file: {file}")
        try:
            A = load_matrix(file)
        except Exception as e:
            print(f"Error loading file {file}:", e)
            continue
        
        print(f"Matrix shape: {A.shape}")
        # Compute the SVD factors via the A^T A method
        U, SigmaT_Sigma, U1, Sigma1, V1 = svd_via_ata(A)
        
        # Print the results (optional)
        np.set_printoptions(precision=4, suppress=True)
        print("\nU1 (columns corresponding to positive eigenvalues):")
        print(U1)
        print("\nΣ1 (diagonal matrix of singular values):")
        print(Sigma1)
        print("\nV1 (computed as A U1 Σ1^{-1}):")
        print(V1)
        
        # Generate a base name for the output files (remove directory and extension)
        base_name = os.path.splitext(os.path.basename(file))[0]
        save_results(base_name, U1, Sigma1, V1, save_dir)
        
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
