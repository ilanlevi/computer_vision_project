import cv2
import numpy as np

from consts import CANNY_SIGMA


def resize_image_and_landmarks(image, landmarks, new_size=None, inter=cv2.INTER_AREA):
    """
    Resize image
    :param image: the image
    :param landmarks: the image landmarks points
    :param new_size: new squared image shape
    :param inter: cv2 interpolation
    :return: tuple of: (resized_image, resized_landmarks)
    """

    if new_size is None:
        return image, landmarks

    # get ratio
    original_shape = image.shape
    ratio_y, ratio_x = (new_size / float(original_shape[0])), (new_size / float(original_shape[1]))

    # resize the image
    resized = cv2.resize(image, (new_size, new_size), interpolation=inter)

    # resize landmarks
    resized_landmarks = np.zeros(landmarks.shape)
    resized_landmarks[:, 0] = landmarks[:, 0] * ratio_y
    resized_landmarks[:, 1] = landmarks[:, 1] * ratio_x

    # return the resized image
    return resized, resized_landmarks


def auto_canny(image, sigma=CANNY_SIGMA):
    """
    Apply canny filter in image
    :param image: the image
    :param sigma: sigma threshold factor
    :return: the canny edge detected image
    """
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    image = image.astype(np.uint8)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def wrap_roi(image, pts):
    """
    Create a roi image from image and landmark.
    Instead of cropping and changing the image size, will change the non roi pixels to 0.
    (for multi-face images)
    :param image: the image
    :param pts: the landmark point
    :return: new image
    """
    x, y, w, h = cv2.boundingRect(pts)
    w = int(1.2 * w)
    h = int(1.2 * w)
    x2, y2 = x + w, y + h

    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])
    x = max(x - w, 0)
    y = max(y - h, 0)

    new_image = np.zeros(image.shape)

    for i in range(y, y2):
        for j in range(x, x2):
            new_image[i, j] = image[i, j]

    return new_image


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):
    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right - >pitch
    x1 = size * pitch + tdx
    y1 = size * pitch + tdy

    # Y-Axis -> yaw
    #        v
    x2 = size * yaw + tdx
    y2 = size * yaw + tdy

    # Z-Axis -> roll
    x3 = size * roll + tdx
    y3 = size * roll + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (255, 255, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (255, 255, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 255, 255), 2)

    return img
