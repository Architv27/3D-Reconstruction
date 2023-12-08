import numpy as np
import cv2 

def ssd(a, b):
    """Compute Sum of Squared Differences (SSD) of two arrays."""
    return np.sum((a - b) ** 2)
def get_disparity(im1, im2, maxDisp, windowSize):
    """
    Compute the disparity map between two stereo images.
    Args:
        im1: First input image.
        im2: Second input image.
        maxDisp: Maximum disparity to consider.
        windowSize: Size of the window used for disparity calculation.
    Returns:
        Disparity map as a numpy array.
    """
    # Convert images to float for precision
    float_im1 = im1.astype(np.float64)
    float_im2 = im2.astype(np.float64)

    # Initialize disparity map and minimum disparity score
    disparity_map = np.zeros_like(float_im1)
    min_disp_score = np.full_like(float_im1, np.inf)

    # Create an averaging mask
    averaging_mask = np.ones((windowSize, windowSize))

    # Initialize the disparity value
    disp = 0

    # Calculate disparity using a while loop
    while disp <= maxDisp:
        # Translate the second image by the disparity value
        shifted_im2 = np.roll(float_im2, disp, axis=1)
        shifted_im2[:, :disp] = 255

        # Compute the squared difference
        squared_diff = cv2.filter2D((float_im1 - shifted_im2) ** 2, -1, averaging_mask, borderType=cv2.BORDER_CONSTANT)

        # Update the disparity map with the new minimum disparities
        update_mask = squared_diff < min_disp_score
        disparity_map[update_mask] = disp
        min_disp_score = np.minimum(min_disp_score, squared_diff)

        # Increment the disparity value
        disp += 1

    return disparity_map
