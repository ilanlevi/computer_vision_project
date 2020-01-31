import cv2


def clean_noise(image):
    """
    pre-process image.
    Runs : median blur and histograms equalization on image
    :param image: the image
    :return: filtered image
    """
    dst = cv2.medianBlur(image, 7)
    dst = cv2.equalizeHist(dst)

    return dst
