import cv2


def read_colour_image(filepath):
    image = cv2.imread(filepath)

    # cv2 reads images in BGR order by default
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # cv2 reads PNG files as uint8 by default which can cause issues when computing negative
    # values, so cast the array to int
    image = image.astype(int)

    return image


def read_grayscale_image(filepath):
    image = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
    image = image.astype(int)
    return image


def write_image(filepath, image):
    cv2.imwrite(filepath, image)
