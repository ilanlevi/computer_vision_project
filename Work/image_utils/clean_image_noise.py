import cv2
import numpy as np


def clean_noise(image, d_type=float):
    """
    pre-process image.
    Runs : median blur and histograms equalization on image
    :param image: the image
    :param d_type:
    :return: filtered image
    """
    dst = np.float32(image)
    dst = cv2.medianBlur(dst, 5)
    dst = _equalize_hist(dst)
    dst = np.asarray(dst)

    return dst.astype(d_type)


def _equalize_hist(img):
    """
    use cv2's equalizeHist in image
    """
    if len(img.shape) == 2:
        dst = np.uint8(img)
        dst = cv2.equalizeHist(dst)
        return dst

    channels = cv2.split(img)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv2.equalizeHist(np.uint8(ch)))
    eq_image = cv2.merge(eq_channels)

    return eq_image
