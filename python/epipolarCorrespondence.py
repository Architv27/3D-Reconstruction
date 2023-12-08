import numpy as np
import cv2

def epipolarCorrespondence(im1, im2, F, pts1):
    """
    Args:
        im1:    Image 1
        im2:    Image 2
        F:      Fundamental Matrix from im1 to im2
        pts1:   coordinates of points in image 1
    Returns:
        pts2:   coordinates of points in image 2
    """
    pts1_homogeneous = np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T

    # Calculate epipolar lines using the fundamental matrix
    epipolar_lines = F @ pts1_homogeneous
    epipolar_lines /= np.abs(epipolar_lines[1, :])  # Normalize the lines

    # Convert pts1 to integer for pixel indexing
    pts1_rounded = np.round(pts1_homogeneous).astype(int)

    # Extract the first point and its patch in im1
    point_x, point_y = pts1_rounded[0, 0], pts1_rounded[1, 0]
    patch_im1 = im1[point_y - 3:point_y + 4, point_x - 3:point_x + 4, :]

    # Define the search boundaries in im2
    search_start = max(0, pts1_rounded[0] - 10)
    search_end = min(im1.shape[1], pts1_rounded[0] + 10)

    # Initialize variables for finding the best match
    best_match = [pts1_rounded[0], int(epipolar_lines[0] * pts1_rounded[0] + epipolar_lines[2])]
    smallest_distance = np.inf

    # Iterate over the search range to find the best match
    for x_coord in range(int(search_start), int(search_end)):
        y_coord = int(epipolar_lines[0] * x_coord + epipolar_lines[2])
        candidate_point = [x_coord, y_coord]

        # Ensure the candidate patch is within image boundaries
        if 0 <= y_coord - 3 < y_coord + 4 <= im2.shape[0] and 0 <= x_coord - 3 < x_coord + 4 <= im2.shape[1]:
            patch_im2 = im2[y_coord - 3:y_coord + 4, x_coord - 3:x_coord + 4]
            distance = np.linalg.norm(patch_im2 - patch_im1)

            # Update the best match if a closer match is found
            if distance < smallest_distance:
                smallest_distance = distance
                best_match = candidate_point

    return np.array(best_match).reshape(1, -1)
