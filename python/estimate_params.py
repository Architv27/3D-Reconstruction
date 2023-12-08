import numpy as np
from scipy.linalg import qr, svd

def estimate_params(P):
    """
    Estimates the camera intrinsic parameters (K), rotation matrix (R), and translation vector (t) from the camera matrix P.

    Args:
        P: Camera matrix with shape [3, 4].

    Returns:
        K: Intrinsic matrix.
        R: Rotation matrix.
        t: Translation vector.
    """

    # Singular Value Decomposition (SVD) of the camera matrix P
    _, _, V = np.linalg.svd(P)
    camera_center = V[-1, :3] / V[-1, -1]  # Normalizing the camera center coordinates

    # Extracting matrix M from P to find intrinsic (K) and rotation (R) matrices
    M = P[:, :3]
    Q_matrix, R_matrix = np.linalg.qr(np.linalg.inv(M))

    # Computing the intrinsic matrix K
    K = np.linalg.inv(R_matrix)
    sign_correction = np.diag(np.sign(np.diag(K)))  # Correcting the sign for the diagonal of K
    K = K @ sign_correction

    # Computing the rotation matrix R
    R = np.linalg.inv(Q_matrix)
    
    # Computing the translation vector t
    t = -R @ camera_center.reshape(-1, 1)

    return K, R, t