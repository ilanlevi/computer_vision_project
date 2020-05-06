from math import cos, sin

import cv2
import numpy as np

from consts import OFFSET_ROI, LANDMARKS_FILE_SUFFIX, LANDMARKS_SHAPE
from my_utils import get_prefix
from .fpn_wrapper import rotation_2_euler


def resize_image_and_landmarks(image, landmarks, new_size=None, inter=cv2.INTER_AREA):
    """
    Resize image
    :param image: the image
    :param landmarks: the image landmarks points
    :param new_size: new squared image shape
    :param inter: cv2 interpolation
    :return: tuple of: (resized_image, resized_landmarks - divided by image size)
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

    resized_landmarks[:, 0] = landmarks[:, 0] * ratio_x
    resized_landmarks[:, 1] = landmarks[:, 1] * ratio_y

    # return the resized image
    return resized, resized_landmarks


def load_image_landmarks(image_path, landmarks_suffix=LANDMARKS_FILE_SUFFIX):
    """
    :param image_path: full path to image
    :return: image landmarks as np array
    :param landmarks_suffix: the landmark file suffix
    :exception ValueError: When cannot find landmarks file for image
    :return: the landmarks from file
    """
    prefix = get_prefix(image_path)
    path = prefix + landmarks_suffix

    _, landmarks = cv2.face.loadFacePoints(path)
    if landmarks is None or []:
        raise ValueError("Cannot file landmarks for: " + image_path)

    landmarks = np.asarray(landmarks)
    landmarks = np.reshape(landmarks, LANDMARKS_SHAPE)

    return landmarks


def create_numbered_image(image, landmarks):
    """
    FOR TESTING ONLY!
    creates landmark only numbered image
    :param image: the original image
    :param landmarks: the image landmark array
    :return: the original image with numbered landmarks on the face
    """
    for n, point in enumerate(landmarks):
        pos = (int(point[0]), int(point[1]))
        label = "%d" % n
        image = cv2.putText(image, label, pos, 0, 0.2, (116, 90, 53), 1, cv2.LINE_AA)

    return image


def wrap_roi(image, pts, offset_roi=OFFSET_ROI):
    """
    Create a roi image from image and landmark. Instead of cropping and changing the image size, it will do something
    like an perspective transformation without changing the face pose (for multi-face images)
    :param offset_roi: offset to pad (ratio of roi) should be: [0, 1]
    :param image: the image
    :param pts: the landmark point
    :return: new image
    """
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    x_max = x_max * (1 + offset_roi)
    y_max = y_max * (1 + offset_roi)
    x_min = x_min * (1 - offset_roi)
    y_min = y_min * (1 - offset_roi)

    x_max = int(min(x_max, image.shape[1] - 1))
    y_max = int(min(y_max, image.shape[0] - 1))
    x_min = int(max(x_min, 0))
    y_min = int(max(y_min, 0))

    # crop extended roi
    if len(image.shape) is 3:
        result = image[y_min:y_max, x_min:x_max, :]
    else:
        result = image[y_min:y_max, x_min:x_max]

    result = np.asarray(result)

    return result


def draw_landmarks_axis(img, pose6DoF, size=50):
    [x, y, z] = rotation_2_euler(pose6DoF[:3])
    q = draw_axis(img, x, -y, z, size=size)
    return q


def create_numbered_mask(landmarks, image_shape):
    """
    FOR TESTING ONLY!
    creates landmark only numbered image
    (similar to create_numbered_image)
    :param landmarks: the image landmark array
    :param image_shape: the output mask size (2d array)
    :return: the landmark image mask
    """
    landmarks_mask = np.zeros(image_shape, dtype=np.float)

    for n, point in enumerate(landmarks):
        pos = (int(point[0]), int(point[1]))
        label = "%d" % n
        cv2.putText(landmarks_mask, label, pos, 0, 0.2, (116, 90, 53), 1, cv2.LINE_AA)

    return landmarks_mask


def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size=50):
    """
    Draw pitch, yaw, roll axis on image
    """
    # yaw = -yaw

    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)

    return img
