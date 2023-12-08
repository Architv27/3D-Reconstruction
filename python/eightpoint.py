import numpy as np
from numpy.linalg import svd
from refineF import refineF

def eightpoint(pts1, pts2, M):
    """
    eightpoint:
        pts1 - Nx2 matrix of (x,y) coordinates
        pts2 - Nx2 matrix of (x,y) coordinates
        M    - max(imwidth, imheight)
    """
    
    # Implement the eightpoint algorithm
    # Generate a matrix F from correspondence '../data/some_corresp.npy'
    # Normalize points
    # Normalize points
    # Define the transformation matrix T
    scaling_factor = 1 / M

    # Transformation matrix
    transform_matrix = np.array([
        [scaling_factor, 0, 0],
        [0, scaling_factor, 0],
        [0, 0, 1]
    ])

    # Homogenizing points
    homogenized_pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    homogenized_pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))

    # Applying the transformation
    transformed_pts1 = homogenized_pts1 @ transform_matrix.T
    transformed_pts2 = homogenized_pts2 @ transform_matrix.T

    # Initializing the A matrix for SVD
    num_points = transformed_pts1.shape[0]
    matrix_A = np.zeros((num_points, 9))

    # Building the A matrix
    for row in range(3):
        for col in range(3):
            idx = (row * 3) + col
            matrix_A[:, idx] = transformed_pts1[:, row] * transformed_pts2[:, col]

    # Singular Value Decomposition
    _, _, matrix_V = np.linalg.svd(matrix_A)
    fundamental_matrix = matrix_V[-1].reshape(3, 3)

    # Enforcing rank 2 constraint
    U, singular_values, Vt = np.linalg.svd(fundamental_matrix)
    singular_values = np.diag(singular_values)
    singular_values[-1, -1] = 0
    fundamental_matrix = U @ singular_values @ Vt

    # Refining the fundamental matrix
    fundamental_matrix = refineF(fundamental_matrix, transformed_pts1, transformed_pts2)

    # Adjusting with the transformation matrix
    final_F = transform_matrix.T @ fundamental_matrix @ transform_matrix

    return final_F
