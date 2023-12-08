import numpy as np
import cv2
from eightpoint import eightpoint
from epipolarCorrespondence import epipolarCorrespondence
from essentialMatrix import essentialMatrix
from camera2 import camera2
from triangulate import triangulate
from epipolarMatchGUI import epipolarMatchGUI
import matplotlib.pyplot as plt
import os

# Loading images and corresponding points
image1 = cv2.imread('../data/im1.png')
image2 = cv2.imread('../data/im2.png')
correspondence_points = np.load('../data/someCorresp.npy', allow_pickle=True).tolist()
image1_pts = correspondence_points['pts1']
image2_pts = correspondence_points['pts2']
normalization_factor = correspondence_points['M']

# Default rotation and translation matrices
rotation_matrix1, translation_vector1 = np.eye(3), np.zeros((3, 1))
rotation_matrix2, translation_vector2 = np.eye(3), np.zeros((3, 1))

# Saving initial extrinsic parameters for reconstruction
np.save('../results/extrinsics', {'R1': rotation_matrix1, 't1': translation_vector1, 'R2': rotation_matrix2, 't2': translation_vector2})

# Fundamental matrix computation
fundamental_mat = eightpoint(image1_pts, image2_pts, normalization_factor)
epipolarMatchGUI(image1, image2, fundamental_mat)

# Loading temple coordinates and finding epipolar correspondences
temple_coords = np.load('../data/templeCoords.npy', allow_pickle=True).tolist()
temple_image1_pts = temple_coords['pts1']
temple_image2_pts = np.zeros_like(temple_image1_pts)

for idx in range(temple_image1_pts.shape[0]):
    temple_image2_pts[idx] = epipolarCorrespondence(image1, image2, fundamental_mat, np.array([temple_image1_pts[idx]]))

# Loading intrinsic camera parameters and computing the essential matrix
camera_intrinsics = np.load('../data/intrinsics.npy', allow_pickle=True).tolist()
intrinsics_matrix1 = camera_intrinsics['K1']
intrinsics_matrix2 = camera_intrinsics['K2']
essential_mat = essentialMatrix(fundamental_mat, intrinsics_matrix1, intrinsics_matrix2)

# Computing the first camera projection matrix and candidates for the second
projection_matrix1 = intrinsics_matrix1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
projection_matrix2_candidates = camera2(essential_mat)

# Triangulating points to find the best projection matrix
minimum_distance = 1e12

for idx in range(4):
    candidate_matrix = projection_matrix2_candidates[:, :, idx]
    if np.linalg.det(candidate_matrix[:3, :3]) != 1:
        candidate_matrix = intrinsics_matrix2 @ candidate_matrix

    triangulated_pts = triangulate(projection_matrix1, temple_image1_pts, candidate_matrix, temple_image2_pts)
    projected_pts1 = projection_matrix1 @ np.hstack((triangulated_pts, np.ones((triangulated_pts.shape[0], 1)))).T
    projected_pts2 = candidate_matrix @ np.hstack((triangulated_pts, np.ones((triangulated_pts.shape[0], 1)))).T

    epsilon_threshold = 1e-6
    projected_pts1[:, projected_pts1[2, :] > epsilon_threshold] /= projected_pts1[2, projected_pts1[2, :] > epsilon_threshold]
    projected_pts2[:, projected_pts2[2, :] > epsilon_threshold] /= projected_pts2[2, projected_pts2[2, :] > epsilon_threshold]

    if np.all(triangulated_pts[:, 2] > 0):
        error_distance1 = np.linalg.norm(temple_image1_pts - projected_pts1[:2, :].T) / triangulated_pts.shape[0]
        error_distance2 = np.linalg.norm(temple_image2_pts - projected_pts2[:2, :].T) / triangulated_pts.shape[0]
        total_error_distance = error_distance1 + error_distance2

        if total_error_distance < minimum_distance:
            minimum_distance = total_error_distance
            final_3d_pts = triangulated_pts
            projection_matrix2 = candidate_matrix

print(f'Minimum error for Image 1 points: {error_distance1}')
print(f'Minimum error for Image 2 points: {error_distance2}')

# Visualizing the 3D points
three_d_plot = plt.figure()
three_d_axis = three_d_plot.add_subplot(111, projection='3d')
three_d_axis.scatter(final_3d_pts[:, 0], final_3d_pts[:, 1], final_3d_pts[:, 2], color='black', marker='.')
three_d_axis.set_box_aspect([1,1,1])
plt.show()

# Computing rotation and translation matrices
rotation_matrix1, _, _, _ = np.linalg.lstsq(intrinsics_matrix1, projection_matrix1[:3, :3], rcond=None)
translation_vector1 = np.linalg.lstsq(intrinsics_matrix1, projection_matrix1[:, 3], rcond=None)[0]
rotation_matrix2, _, _, _ = np.linalg.lstsq(intrinsics_matrix2, projection_matrix2[:3, :3], rcond=None)
translation_vector2 = np.linalg.lstsq(intrinsics_matrix2, projection_matrix2[:, 3], rcond=None)[0]

# Ensuring the correct shape of rotation and translation matrices
rotation_matrix1 = rotation_matrix1.T
translation_vector1 = translation_vector1.ravel()
rotation_matrix2 = rotation_matrix2.T
translation_vector2 = translation_vector2.ravel()

# Saving final extrinsic parameters
os.makedirs('../results/extrinsics', exist_ok=True)
np.save('../results/extrinsics', {'R1': rotation_matrix1, 't1': translation_vector1, 'R2': rotation_matrix2, 't2': translation_vector2})

print("Extrinsics saved successfully.")

