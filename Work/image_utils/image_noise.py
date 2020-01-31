import cv2
import numpy as np


def clean_noise(image):
    """
    pre-process image.
    Runs : median blur and histograms equalization on image
    :param image: the image
    :return: filtered image
    """
    dst = np.float32(image)
    dst = cv2.medianBlur(dst, 5)
    dst = np.uint8(dst)
    dst = cv2.equalizeHist(dst)

    return dst
