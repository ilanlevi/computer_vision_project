import numpy as np

from consts import FACIAL_LANDMARKS_68_IDXS_FLIP, L_EYE_IDX, R_EYE_IDX


def flip_landmarks(landmarks_points, width):
    """
    if a horizontal flip happens we to flip the target x coordinates accordingly.
    :param landmarks_points: the landmarks array
    :param width: the original image width
    :return: landmarks_points after flip
    """
    landmarks_points[:, 0] = width - landmarks_points[:, 0]

    # Creating flipped landmarks with new indexing
    landmarks_points_flipped = np.zeros((68, 2))
    for i in range(len(FACIAL_LANDMARKS_68_IDXS_FLIP)):
        landmarks_points_flipped[i, 0] = landmarks_points[FACIAL_LANDMARKS_68_IDXS_FLIP[i] - 1, 0]
        landmarks_points_flipped[i, 1] = landmarks_points[FACIAL_LANDMARKS_68_IDXS_FLIP[i] - 1, 1]

    return landmarks_points_flipped


def flip_if_needed(landmarks_points, width, flip_horizontal=False):
    """
    if a horizontal flip happens we to flip the target x coordinates accordingly.
    Also flips if the x of right eye is bigger then x of left eye
    :param width: image width
    :param flip_horizontal: was flipped randomly
    :param landmarks_points: the landmarks array
    :return: landmarks_points after flipped if needed or the original landmarks_points
    """
    if flip_horizontal or landmarks_points[R_EYE_IDX, 0] > landmarks_points[L_EYE_IDX, 0]:
        landmarks_points = flip_landmarks(landmarks_points, width)
    return landmarks_points


def pose_transform(pose_vector, transform_parameters):
    """
    Randomly transform face pose
    :param pose_vector: the original pose vector
    :param transform_parameters: the random transformation params
    :return: new pose vector after transformation
    """
    transform_matrix = _get_affine_transform_matrix(0, 0, transform_parameters.get('theta', 0),
                                                    transform_parameters.get('tx', 0),
                                                    transform_parameters.get('ty', 0),
                                                    transform_parameters.get('shear', 0),
                                                    transform_parameters.get('zx', 1),
                                                    transform_parameters.get('zy', 1), use_offset=False)
    if transform_matrix is None:
        return pose_vector

    pose_vector = np.dot(transform_matrix, pose_vector)
    pose_vector = pose_vector.T

    return pose_vector


def get_affine_transform_matrix(transform_parameters):
    """
    create affine transformation matrix - without considering the distance (without translation center)
    :param transform_parameters: the params
    :return:  transformation matrix - if all are 0, will return None
    """
    transform_matrix = _get_affine_transform_matrix(0, 0, transform_parameters.get('theta', 0),
                                                    transform_parameters.get('tx', 0),
                                                    transform_parameters.get('ty', 0),
                                                    transform_parameters.get('shear', 0),
                                                    transform_parameters.get('zx', 1),
                                                    transform_parameters.get('zy', 1), use_offset=False)
    return transform_matrix


def _get_affine_transform_matrix(height, width, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1, use_offset=True):
    """
    Compute the affine transformation specified by the parameters given.

    :param height: Height of the image to transform
    :param width: Width of the image to transform
    :param theta: Rotation angle in degrees.
    :param tx: Width shift.
    :param ty: Height shift.
    :param shear: Shear angle in degrees.
    :param zx: Zoom in x direction.
    :param zy: Zoom in y direction
    :param use_offset: add translation or not (with center ratio)

    :return: The affine transformation matrix for the parameters or None if no transformation is needed.
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

    if transform_matrix is not None and use_offset:
        transform_matrix = _transform_matrix_offset_center(
            transform_matrix, height, width)

    return transform_matrix


def _transform_matrix_offset_center(matrix, y, x):
    o_x = float(x) / 2
    o_y = float(y) / 2
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
