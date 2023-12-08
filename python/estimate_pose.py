import numpy as np
from scipy.linalg import svd

def estimate_pose(x, X):
    """
    Estimates the camera matrix P given 2D and 3D points.

    Args:
        x: 2D points with shape [2, N]
        X: 3D points with shape [3, N]

    Returns:
        P: Estimated camera matrix with shape [3, 4]
    """
    # Number of points
    N = x.shape[1]

    # Construct matrix A for DLT; A * p = 0
    A = np.zeros((2 * N, 12))
    for i in range(N):
        X_i = X[:, i]
        x_i = x[:, i]
        A[2 * i] = np.array([-X_i[0], -X_i[1], -X_i[2], -1, 0, 0, 0, 0, x_i[0] * X_i[0], x_i[0] * X_i[1], x_i[0] * X_i[2], x_i[0]])
        A[2 * i + 1] = np.array([0, 0, 0, 0, -X_i[0], -X_i[1], -X_i[2], -1, x_i[1] * X_i[0], x_i[1] * X_i[1], x_i[1] * X_i[2], x_i[1]])

    # Solve for p using SVD
    U, S, Vt = svd(A)
    p = Vt[-1]

    # Reshape p into a 3x4 matrix
    P = p.reshape(3, 4)

    return P
