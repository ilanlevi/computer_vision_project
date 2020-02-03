import cv2
import numpy as np

from consts import LANDMARKS_FILE_SUFFIX, LANDMARKS_SHAPE, FACIAL_LANDMARKS_68_IDXS_FLIP
from my_utils.my_io import get_prefix


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


def _flip(landmarks_points, height, width, flip_horizontal, flip_vertical):
    """
    if a horizontal flip happens we to flip the target x coordinates accordingly.
    :param landmarks_points: the landmarks array
    :return: landmarks_points after flipped if needed or the original landmarks_points
    """
    if flip_horizontal or flip_vertical:
        landmarks_points = landmarks_points * np.array(
            [1 - 2 * flip_horizontal, 1 - 2 * flip_vertical]) + np.array(
            [width * flip_horizontal, height * flip_vertical])
    if flip_horizontal:
        # flip x's of points
        for a, b in FACIAL_LANDMARKS_68_IDXS_FLIP:
            i, j = landmarks_points[b, 0], landmarks_points[b, 1]
            landmarks_points[b, 0], landmarks_points[b, 1] = landmarks_points[a, 0], landmarks_points[a, 1]
            landmarks_points[a, 0], landmarks_points[a, 1] = i, j
    return landmarks_points


def create_numbered_mask(landmarks, image_shape):
    """
    FOR TESTING ONLY!
    creates landmark only numbered image
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


def landmarks_transform(width, height, landmarks, transform_parameters):
    """Applies a transformation to an tensor of landmarks on an image according to given parameters.
    # Arguments
        x: 3D tensor, single image. Required only to analyze image dimensions.
        landmarks: 3D tensor, containing all the 2D points to be transformed.
        transform_parameters: Dictionary with string - parameter pairs
            describing the transformation.
            Currently, the following parameters
            from the dictionary are used:
            - `'theta'`: Float. Rotation angle in degrees.
            - `'tx'`: Float. Shift in the x direction.
            - `'ty'`: Float. Shift in the y direction.
            - `'shear'`: Float. Shear angle in degrees.
            - `'zx'`: Float. Zoom in the x direction.
            - `'zy'`: Float. Zoom in the y direction.
            - `'flip_horizontal'`: Boolean. Horizontal flip.
            - `'flip_vertical'`: Boolean. Vertical flip.
    # Returns
        A transformed version of the landmarks.
    """
    landmarks = _affine_transform_points(landmarks, height, width,
                                         transform_parameters.get('theta', 0),
                                         transform_parameters.get('tx', 0),
                                         transform_parameters.get('ty', 0),
                                         transform_parameters.get('shear', 0),
                                         transform_parameters.get('zx', 1),
                                         transform_parameters.get('zy', 1)
                                         )

    # check if flip happened
    landmarks = _flip(
        landmarks,
        height,
        width,
        flip_horizontal=transform_parameters.get('flip_horizontal', 0),
        flip_vertical=transform_parameters.get('flip_vertical', 0)
    )

    return landmarks


def _affine_transform_points(points, height, width, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1):
    """Applies an affine transformation of the points specified
     by the parameters given.
    # Arguments
        points: 3D tensor, containing all the 2D points to be transformed.
        height: Height of the image the points are part of
        width: Width of the image the points are part of
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Height shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
    # Returns
        The transformed version of the points.
    """
    transform_matrix = _get_affine_transform_matrix(
        height, width,
        theta, tx, ty, shear, zx, zy)

    if transform_matrix is not None:
        homogeneous_points = np.transpose(points)
        homogeneous_points = np.insert(homogeneous_points[[1, 0]], 2, 1, axis=0)
        inverse = np.linalg.inv(transform_matrix)
        homogeneous_points = np.dot(inverse, homogeneous_points)
        points = homogeneous_points[[1, 0]]
        points = np.transpose(points)

    return points


def _transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2
    o_y = float(y) / 2
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def _get_affine_transform_matrix(height, width, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1):
    """Compute the affine transformation specified by the parameters given.

    # Arguments
        height: Height of the image to transform
        width: Width of the image to transform
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Height shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction

    # Returns
        The affine transformation matrix for the parameters
        or None if no transformation is needed.
    """

    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        transform_matrix = _transform_matrix_offset_center(
            transform_matrix, height, width)

    return transform_matrix
