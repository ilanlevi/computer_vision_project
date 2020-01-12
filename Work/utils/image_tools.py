import time

import cv2
import numpy as np
import math


def load_images(images_path_list, width=None, height=None, gray=True, print_data=False):
    """
    for each image: resize image and convert to gray scale
    :param print_data: print duration data (default is false)
    :param gray: convert to gray or not
    :param height: the height
    :param images_path_list: images list (full path) to load
    :param width: the width
    :return: image list
    """

    start = time.time()

    ims = []
    for image_path in images_path_list:
        try:
            if gray:
                im = cv2.imread(image_path, 1)
            else:
                im = cv2.imread(image_path)
            im = resize(im, width, height)
            ims.append(im)
        except Exception as e:
            print ('Error while reading image!Path= %s\nError= %s' % (image_path, str(e)))

    if print_data:
        print 'Loading images took: %.2f seconds' % (time.time() - start)

    return ims


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize image
    :param image: the image
    :param width: new width
    :param height: new height
    :param inter: cv2 interpolation
    :return: the resized image
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def matrix2angle(R):
    """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
    Args:
        :param R: (3,3). rotation matrix
    :returns:
        x: yaw
        y: pitch
        z: roll
    """

    if R[2, 0] != 1 or R[2, 0] != -1:
        x = -math.asin(R[2, 0])
        # x = np.pi - x
        y = math.atan2(R[2, 1] / math.cos(x), R[2, 2] / math.cos(x))
        z = math.atan2(R[1, 0] / math.cos(x), R[0, 0] / math.cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + math.atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + math.atan2(-R[0, 1], -R[0, 2])

    return x, y, z


def P2sRt(P):
    """
    decompositing camera matrix P.
    :param P: (3, 4). Affine Camera Matrix.
    :returns
        s: scale factor.
        R: (3, 3). rotation matrix.
        t3d: (3,). 3d translation.
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r2, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def roi_from_landmarks(image, face_landmarks, height=250, width=250, d_type="int"):
    shape = shape_to_np(face_landmarks, d_type)
    (x, y, w, h) = cv2.boundingRect(np.array(shape))
    roi = image[y:y + h, x:x + w]
    roi = resize(roi, height, width)

    return roi


def get_camarx_matrix(image, landmarks, height=250, width=250, d_type="int"):
    shape = np.shape(landmarks)
    lmarks = np.reshape(landmarks, (shape[0], shape[1]))
    lmarks = lmarks.astype(d_type)

    (x, y, w, h) = cv2.boundingRect(np.array([lmarks]))

    roi = image[y:y + h, x:x + w]
    roi = resize(roi, height, width, inter=cv2.INTER_CUBIC)

    size = np.shape(roi)

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)

    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    return camera_matrix, roi
