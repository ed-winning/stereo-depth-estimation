import numpy as np

from stereo import (compute_disparity_image, compute_disparity_score,
                    visualize_disparity)
from utils import read_colour_image, read_grayscale_image, write_image

if __name__ == "__main__":
    Il = read_colour_image("images/cones_image_02.png")
    Ir = read_colour_image("images/cones_image_06.png")
    bbox = np.array([[0, 449], [0, 374]])

    # Id = compute_disparity_image(Il, Ir, bbox)
    Id = read_grayscale_image("images/disparity_estimate.png")
    It = read_grayscale_image("images/cones_disp_02.png")

    rms = compute_disparity_score(It, Id)
    print(f"Disparity Score (RMS): {rms:.2f}")

    write_image("images/disparity_estimate.png", Id)

    visualize_disparity(
        It,
        "Ground Truth Disparity",
        save_visualization=True,
        filepath="images/visualize_cones_disp_02.png",
    )
    visualize_disparity(
        Id,
        "Estimated Disparity",
        save_visualization=True,
        filepath="images/visualize_disparity_estimate.png",
    )
