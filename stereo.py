import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def compute_disparity_image(Il, Ir, bbox, window_size=7):
    """
    Returns a disparity image estimated from two stereo images.
     
    A local, fixed-support stereo correspondence algorithm is used for estimation.
    Specifically, a window-based method using the sum-of-absolute-difference (SAD) similarity
    measure is implemented.

    Parameters:
    Il (numpy.ndarray): left stereo image.
        Shape: (height, width, channels)

    Ir (numpy.ndarray): right stereo image.
        Shape: (height, width, channels)

    bbox (numpy.ndarray): bounding box relative to the left image containing top left corner and
    bottom right corner (inclusive).
        Shape: (2, 2), where the first column is the top left corner and the second column is the
        bottom right corner. Height is v and width is u.

    window_size (int): the size of the square window used to compute SAD.

    Returns:
    (numpy.ndarray): estimated disparity image
        Shape: (v, u)
    """
    x_i = bbox[0, 0]
    x_f = bbox[0, 1]
    y_i = bbox[1, 0]
    y_f = bbox[1, 1]
    u = x_f - x_i + 1
    v = y_f - y_i + 1

    Id = np.zeros((v, u), dtype=int)

    r = window_size // 2
    total_iterations = len(range(y_i + r, y_f - r + 1)) * len(range(x_i + r, x_f - r + 1))
    progress_bar = tqdm(total=total_iterations)
    for y in range(y_i + r, y_f - r + 1):
        for x in range(x_i + r, x_f - r + 1):
            window_left = Il[y - r : y + r + 1, x - r : x + r + 1, :]
            window_right = Ir[y - r : y + r + 1, x - r : x + r + 1, :]

            # Note for `window_left - window_right` in the line below: if the numpy arrays
            # are using unsigned integers then there will be an overflow when the difference
            # is negative. The solution is to cast the numpy array to type int.
            #
            # >>> numpy.uint32(0) - numpy.uint32(1)
            # 4294967295
            minimum_SAD = np.sum(np.abs(window_left - window_right))

            disparity = 0
            for d in range(1, 64):
                if x - r - d < 0:
                    break
                window_right = Ir[y - r : y + r + 1, x - r - d : x + r + 1 - d, :]
                current_SAD = np.sum(np.abs(window_left - window_right))
                if current_SAD < minimum_SAD:
                    minimum_SAD = current_SAD
                    disparity = d
            Id[y - y_i, x - x_i] = disparity
            progress_bar.update(1)
    progress_bar.close()
    return Id


def compute_disparity_score(It, Id):
    mask = It != 0
    N = np.sum(mask)

    if N != 0:
        rms = np.sqrt(np.sum((Id[mask] - It[mask]) ** 2) / N)
    else:
        raise ZeroDivisionError(
            "Invalid ground truth disparity image (i.e. all disparity values are zero)."
        )

    return rms


def view_disparity(image):
    # Adjust image
    contrast = 3
    brightness = 50

    adjusted_image = contrast * image + brightness
    adjusted_image = np.clip(adjusted_image, 0, 255)
    adjusted_image = adjusted_image.astype(np.uint8)

    # Show image
    plt.imshow(adjusted_image, cmap="gray", vmin=0, vmax=255)
    plt.show()
