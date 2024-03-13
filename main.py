import numpy as np

from stereo import (compute_disparity_image, compute_disparity_score,
                    view_disparity)
from utils import read_colour_image, read_grayscale_image, write_image

if __name__ == "__main__":
    Il = read_colour_image("images/cones_image_02.png")
    Ir = read_colour_image("images/cones_image_06.png")
    bbox = np.array([[0, 449], [0, 374]])

    Id = compute_disparity_image(Il, Ir, bbox)
    It = read_grayscale_image("images/cones_disp_02.png")

    rms = compute_disparity_score(It, Id)
    print(f"Disparity Score (RMS): {rms:.2f}")

    view_disparity(Id)
    write_image("images/disparity_estimate.png", Id)
