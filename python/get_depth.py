import numpy as np

def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    """
    creates a depth map from a disparity map (DISPM).
    """
    # Compute baseline (distance between optical centers)
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    b = np.linalg.norm(c1 - c2)

    # Focal length from the K1 matrix
    f = K1[0, 0]

    # Initialize the depth map
    depthM = np.zeros_like(dispM, dtype=float)

    # Wherever disparity is zero, set depth to zero to avoid division by zero
    non_zero_disp = dispM > 0
    depthM[non_zero_disp] = (b * f) / dispM[non_zero_disp]

    return depthM
    # depthM = np.zeros_like(dispM, dtype=float)

    # return depthM

