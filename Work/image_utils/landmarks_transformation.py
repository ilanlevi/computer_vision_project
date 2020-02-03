import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# todo - delete this

class ImageDataGeneratorLandmarks(ImageDataGenerator):
    """
    Generator to transform the landmarks in the same way as the augmented (transformed) images.

    This generator assumes all landmarks belong to images of the same size (width x height).
    If you have annotated landmarks in images of different size but resize them later on,
    for example using the target_size parameter of the ImageDataGenerator, then you must
    first normalize your landmark coordinates to the target image size.
    You can do so through the helper function normalize_landmarks in this module.

    After transforming (rotating, scaling, shifting, ...) landmarks can end up outside visible part of the image.
    This generator automatically calculates if the landmark is still visible. If not, the landmark-present
    indicator is automatically set to 0.0.

    Landmarks must be placed in the 'x' of the dataset and be encoded as follows:
        Shape:
            (batch_index, landmark, 3)

        Where each landmark consists of 3 elements:
            - X coordinate
            - Y coordinate
            - Present indicator (0.0 if not present, 1.0 if present)

        Example:
            [
                [   [landmark_1_x, landmark_1_y, landmark_1_present],
                    [landmark_2_x, landmark_2_y, landmark_2_present] ],
                [   [landmark_1_x, landmark_1_y, landmark_1_present],
                    [landmark_2_x, landmark_2_y, landmark_2_present] ]
            ]

    Code based on:
    https://github.com/keras-team/keras-preprocessing/pull/132/commits
    """

    def __init__(self, width, height,
                 featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                 samplewise_std_normalization=False, zca_whitening=False, zca_epsilon=1e-6, rotation_range=0,
                 width_shift_range=0., height_shift_range=0., brightness_range=None, shear_range=0., zoom_range=0.,
                 channel_shift_range=0., fill_mode='nearest', cval=0., horizontal_flip=False, vertical_flip=False,
                 rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None):
        """
        :type width: Integer. Width of the images.
        :type height: Integer. Height of the images.
        """
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization,
                         samplewise_std_normalization, zca_whitening, zca_epsilon, rotation_range, width_shift_range,
                         height_shift_range, brightness_range, shear_range, zoom_range, channel_shift_range, fill_mode,
                         cval, horizontal_flip, vertical_flip, rescale, preprocessing_function, data_format,
                         validation_split, dtype)
        self.image_width = width
        self.image_height = height

    def flow_landmarks(self, landmarks, batch_size=32, shuffle=True, seed=None, subset=None):
        # Convert landmarks to 4-rank to remain compatible with downstream Keras code
        s = landmarks.shape
        x = landmarks.reshape((s[0], s[1], s[2], 1))
        return super().flow(x, None, batch_size, shuffle, None, seed, None, '', 'png', subset)

    def apply_transform(self, x, transform_parameters):
        # Fix translation!
        # It is miscalculated because is it based the tensor shape is taken as image width/height.
        wrong_img_shape = x.shape
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        tx_factor = self.image_height / wrong_img_shape[img_row_axis]
        ty_factor = self.image_width / wrong_img_shape[img_col_axis]
        transform_parameters['tx'] = transform_parameters['tx'] * tx_factor
        transform_parameters['ty'] = transform_parameters['ty'] * ty_factor

        # Landmarks are of a single sample. So a 3-rank array.

        # Convert to 2-rank
        landmarks = x.reshape((x.shape[0], x.shape[1]))

        # Extract only the X/Y for transformations and transform
        landmarks_xy = landmarks[:, 0:2]
        landmarks_xy_transformed = _transform_landmarks(
            self.image_width,
            self.image_height,
            landmarks_xy,
            transform_parameters
        )

        # Determine if the landmarks are still inside the image
        landmarks_visibilities = (landmarks_xy_transformed[:, 0] >= 0) & \
                                 (landmarks_xy_transformed[:, 0] < self.image_width) & \
                                 (landmarks_xy_transformed[:, 1] >= 0) & \
                                 (landmarks_xy_transformed[:, 1] < self.image_height)

        # Pack result back in original landmarks (they are a copy anyway)
        landmarks[:, 0:2] = landmarks_xy_transformed
        landmarks[:, 2] = landmarks_visibilities

        # Restore the 3-rank array
        return landmarks.reshape(x.shape)


def derank_landmarks_output(landmarks_output):
    """
    Removes the last dimension from the landmarks output by the ImageDataGeneratorLandmarks.

    :param landmarks_output:
    :return:
    """
    return landmarks_output.reshape(landmarks_output.shape[:-1])


def normalize_landmarks(landmarks, image_widths, image_heights, target_width=1.0, target_height=1.0):
    """
    Normalize image landmarks between 0.0 and 1.0 or between 0.0 and target image width/height.

    Input:
        Numpy array of landmarks of shape (n, 2+) where:
            - n is the number of landmarks
            - 2+ are the landmark x, landmark y and optionally other data that is not touched.

    :param landmarks: Numpy array containing landmarks
    :param image_widths: Numpy array of widths of images in same order as the landmarks
    :param image_heights: Numpy array of heights of images in same order as the landmarks
    :param target_width: Float. Optional. Width of target image, default 1.0
    :param target_height: Float. Optional. Height of target image, default 1.0
    :return: Copy of landmarks array with the x and y normalized.
    """
    widths_correction = target_width / image_widths
    heights_correction = target_height / image_heights

    landmarks_normalized = landmarks.copy()
    for i in range(landmarks.shape[1]):
        landmarks_normalized[:, i, 0] = landmarks[:, i, 0] * widths_correction
        landmarks_normalized[:, i, 1] = landmarks[:, i, 1] * heights_correction

    return landmarks_normalized


def _transform_landmarks(width, height, landmarks, transform_parameters):
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

    landmarks = _flip_points(
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


def _flip_points(points, height, width, flip_horizontal=False, flip_vertical=False):
    """Flips the coordinates of points in a frame with dimensions height and width
    horizontally and/or vertically.
    # Arguments
        x: Point tensor. Must be 3D.
        height: Height of the frame.
        width: Width of the frame.
        flip_horizontal: Boolean if coordinates shall be flipped horizontally.
        flip_vertical: Boolean if coordinates shall be flipped vertically.
    # Returns
        Flipped Numpy point tensor.
    """
    if flip_horizontal or flip_vertical:
        points = points * np.array(
            [1 - 2 * flip_horizontal, 1 - 2 * flip_vertical]) + np.array(
            [width * flip_horizontal, height * flip_vertical])
    return points


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


def _transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2
    o_y = float(y) / 2
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix
