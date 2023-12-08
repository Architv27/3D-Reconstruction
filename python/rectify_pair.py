import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    """
    Compute rectification matrices and updated camera parameters for stereo cameras.

    Args:
        K1, K2: Intrinsic matrices of camera 1 and camera 2.
        R1, R2: Rotation matrices of camera 1 and camera 2.
        t1, t2: Translation vectors of camera 1 and camera 2.

    Returns:
        M1, M2: Rectification matrices for camera 1 and camera 2.
        K1n, K2n: Updated intrinsic matrices for camera 1 and camera 2.
        R1n, R2n: Updated rotation matrices for camera 1 and camera 2.
        t1n, t2n: Updated translation vectors for camera 1 and camera 2.
    """

    # Compute the optical centers.
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)

    # Compute the new rotation matrix.
    r1 = (c1- c2).flatten() / np.linalg.norm(c1 - c2)
    r2 = np.cross(R1[2, :], r1)
    r2 /= np.linalg.norm(r2)
    r3 = np.cross(r2, r1)

    # New rotation matrix is composed of the new axes.
    Rn = np.vstack((r1, r2, r3)).T

    # The new intrinsic parameters can be an arbitrary selection
    # Here we simply choose K2 for both cameras
    K1n = K2n = K2
    
    # New translations are the negated optical centers transformed by the new rotation
    t1n = Rn @ c1
    t2n = Rn @ c2
    
    # The rectification matrices are then derived from the intrinsic and rotation matrices
    M1 = K1n @ Rn @ np.linalg.inv(K1)
    M2 = K2n @ Rn @ np.linalg.inv(K2)
    return M1, M2, K1n, K2n, Rn, Rn, t1n, t2n

