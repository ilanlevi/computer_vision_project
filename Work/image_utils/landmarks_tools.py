import cv2
import numpy as np

from consts import R_EYE_IDX, L_EYE_IDX, FACIAL_LANDMARKS_68_IDXS_FLIP, LANDMARKS_FILE_SUFFIX, LANDMARKS_SHAPE
from my_utils.my_io import get_prefix


def load_image_landmarks(image_path, new_image_shape, landmarks_suffix=LANDMARKS_FILE_SUFFIX):
    """
    :param image_path: full path to image
    :param new_image_shape: for rescaling - the original image size
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


def get_landmarks_from_masks(landmarks_images):
    """
    The reverse of create mask -> calculate landmark from mask image.
    Uses _adjust_horizontal_flip function to flip the landmarks indexes if the image was flipped.
    :param landmarks_images: the landmark image masks array
    :return: image landmarks as (68, 2) array
    """

    landmarks_points = []

    for landmarks_image in landmarks_images:
        ix, iy = np.where(landmarks_image > 0)
        if len(ix) == 0:
            return None
        landmarks_points.append([[np.mean(iy), np.mean(ix)]])

    landmarks_points = np.array(landmarks_points)
    landmarks_points = np.reshape(landmarks_points, LANDMARKS_SHAPE)
    landmarks = _adjust_horizontal_flip(landmarks_points)
    return landmarks


def _adjust_horizontal_flip(landmarks_points):
    """
    if a horizontal flip happens we to flip the target coordinates accordingly.
    (We will know that the image was flipped if the right eye is after the left index)
    :param landmarks_points: the landmarks array
    :return: landmarks_points after flipped if needed or the original landmarks_points
    """
    if landmarks_points[R_EYE_IDX][1] > landmarks_points[L_EYE_IDX][1]:  # check if flip happens
        # x-cord of right eye is less than x-cord of left eye
        # horizontal flip happened!
        for a, b in FACIAL_LANDMARKS_68_IDXS_FLIP:
            tmpX, tmpY = landmarks_points[b]
            landmarks_points[b] = landmarks_points[a]
            landmarks_points[a] = [tmpX, tmpY]
    landmarks_points = np.asarray(landmarks_points)
    landmarks_points = np.reshape(landmarks_points, (68, 2))

    return landmarks_points


def create_single_landmark_mask(landmark, image_shape):
    """
    creates the mask landmark image
    :param landmark: image single landmark
    :param image_shape: the output mask size (without channel)
    :return: the landmark image mask
    """
    landmarks_mask = np.zeros((image_shape[0], image_shape[1]))
    landmarks_mask[int(landmark[1]), int(landmark[0])] = 255

    return landmarks_mask


def create_mask_from_landmarks(landmarks, image_shape):
    """
    creates the mask landmark image
    :param landmarks: all of the landmark
    :param image_shape: the output mask size (without channel)
    :return: the landmark image mask
    """
    landmarks_mask = np.zeros((image_shape[1], image_shape[2]))
    for landmark in landmarks:
        landmarks_mask[int(landmark[1]), int(landmark[0])] = 255

    return landmarks_mask


def create_numbered_mask(landmarks, image_shape):
    """
    FOR TESTING ONLY!
    creates landmark only numbered image
    :param landmarks: the image landmark array
    :param image_shape: the output mask size (2d array)
    :return: the landmark image mask
    """
    new_shape = (512, 512)  # the mask size

    shape = (image_shape[0], image_shape[1])

    # calc reverse for mask
    ratio_x = (new_shape[0] / float(shape[0]))
    ratio_y = (new_shape[1] / float(shape[1]))

    # resize landmarks
    landmarks2 = np.array(landmarks.copy())
    landmarks2 = landmarks2.astype(np.float)
    landmarks2[:, 0] = landmarks2[:, 0] * ratio_y
    landmarks2[:, 1] = landmarks2[:, 1] * ratio_x

    landmarks_mask = np.zeros(new_shape, dtype=np.float)

    for n, (x, y) in enumerate(landmarks2):
        label = "%d" % n
        cv2.putText(landmarks_mask, label, (int(x), int(y)), 0, 0.4, (116, 90, 53), 1, cv2.LINE_AA)

    return landmarks_mask
