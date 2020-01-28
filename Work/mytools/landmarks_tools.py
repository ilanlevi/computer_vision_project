import cv2
import numpy as np

from consts import DataSetConsts as dC
from consts import R_EYE, L_EYE, FACIAL_LANDMARKS_68_IDXS_FLIP
from .my_io import get_prefix


def load_image_landmarks(image_path, new_image_shape=None, landmarks_suffix=dC.LANDMARKS_FILE_SUFFIX):
    """
    :param image_path: full path to image
    :exception ValueError: When cannot find landmarks file for image
    :param new_image_shape: for rescaling - the original image size
    :return: image landmarks as np array
    :param landmarks_suffix: the landmark file suffix
    """
    # landmarks = get_landmarks(image_path, self.landmark_suffix)

    prefix = get_prefix(image_path)
    path = prefix + landmarks_suffix

    _, landmarks = cv2.face.loadFacePoints(path)
    if landmarks is None or []:
        raise ValueError("Cannot file landmarks for: " + image_path)
    landmarks = np.asarray(landmarks)
    landmarks = np.reshape(landmarks, (68, 2))
    if new_image_shape is not None:
        image = cv2.imread(image_path)
        original_shape = image.shape
        ratio_x = (new_image_shape[0] / float(original_shape[0]))
        ratio_y = (new_image_shape[1] / float(original_shape[1]))
        # resize landmarks
        landmarks = np.array(landmarks)
        landmarks[:, 0] = landmarks[:, 0] * ratio_y
        landmarks[:, 1] = landmarks[:, 1] * ratio_x

    return landmarks


def get_landmarks_from_mask(landmarks_image):
    """
    :param landmarks_image: the landmark image mask
    :return: image landmarks as np array
    """

    landmarks_points = []
    for i in range(68):
        ix, iy = np.where(landmarks_image == i)
        if len(ix) == 0:
            return None
        landmarks_points.extend([np.mean(iy), np.mean(ix)])

    landmarks_points = np.array(landmarks_points)
    landmarks = _adjust_horizontal_flip(landmarks_points)
    return landmarks


def _adjust_horizontal_flip(landmarks_points):
    """
    if a horizontal flip happens we to flip the target coordinates accordingly
    :param landmarks_points: the landmarks
    :return: landmarks_points after flipped if needed
    """
    if landmarks_points[R_EYE] > landmarks_points[L_EYE]:  # check if flip happens
        # x-cord of right eye is less than x-cord of left eye
        # horizontal flip happened!
        for a, b in FACIAL_LANDMARKS_68_IDXS_FLIP:
            landmarks_points[a], landmarks_points[b] = (landmarks_points[b], landmarks_points[a])
    return landmarks_points


def create_landmark_mask(landmarks, image_shape):
    """
    creates the mask landmark image and saves if wished
    :param landmarks: the image landmark
    :param image_shape: the output mask size (image_size, image_size)
    :return: the landmark image mask
    """
    shape = (image_shape[0], image_shape[1])

    landmarks_mask = np.zeros(shape)
    landmarks_mask[:] = -1

    for index, (ix, iy) in enumerate(landmarks):
        landmarks_mask[int(iy), int(ix)] = index

    return landmarks_mask
