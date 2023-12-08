import numpy as np

def triangulate(P1, pts1, P2, pts2):
    """
    Estimate the 3D positions of points from 2d correspondence
    Args:
        P1:     projection matrix with shape 3 x 4 for image 1
        pts1:   coordinates of points with shape N x 2 on image 1
        P2:     projection matrix with shape 3 x 4 for image 2
        pts2:   coordinates of points with shape N x 2 on image 2

    Returns:
        Pts3d:  coordinates of 3D points with shape N x 3
    """
    num_points = pts1.shape[0]
    pts3d = np.zeros((num_points, 3))

    for i in range(num_points):
        # Extracting the coordinates from each set of points
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        # Constructing matrix A for each correspondence
        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :]
        ])

        # Solve for X: AX = 0 using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]

        # Homogeneous to Cartesian coordinate conversion
        pts3d[i] = X[:3] / X[3]

    return pts3d
