import numpy as np
import cv2
from utils import resize
import copy


def shape_to_np(shape, d_type="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((len(shape), 2), dtype=d_type)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, len(shape)):
        coords[i] = (shape[i][0], shape[i][1])

    # return the list of (x, y)-coordinates
    return coords


def roi_from_landmarks(image, face_landmarks, f=50, d_type="int"):
    shape = np.shape(face_landmarks)
    lmarks = np.reshape(face_landmarks, (shape[0], shape[1]))
    lmarks = lmarks.astype(d_type)

    (x, y, w, h) = cv2.boundingRect(np.array([lmarks]))

    y = y - (f / 2)
    x = x - (f / 2)

    h = h + f
    w = w + f

    roi = image[y:y + h, x:x + w]
    # roi = np.asarray(roi, dtype=d_type)

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


def display_landmarks(img, landmarks, name='', show=False):
    # loop over the face detections
    shape = np.shape(landmarks)
    lndmarks = np.reshape(landmarks, (shape[0], shape[1]))

    for (x, y) in lndmarks:
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        x = int(x)
        y = int(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks
    if show:
        cv2.imshow("Output - %s" % name, img)
    return img


def display_bbox(img, lndmarks, name='', factor=0.2):
    shape = np.shape(img)
    bbox_add = (shape[0] + shape[1]) * factor / 2

    # loop over the face detections
    lefX = int(np.min(lndmarks[:][0]) - bbox_add)
    rightX = int(np.max(lndmarks[:][0]) + bbox_add)
    upY = int(np.min(lndmarks[:][1]) - bbox_add)
    downY = int(np.max(lndmarks[:][1]) + bbox_add)

    cv2.rectangle(img, (lefX, upY), (rightX, downY), (0, 20, 200), 10)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output - %s" % name, img)


def draw_axis_on_image(image, rx, ry, rz, tx, ty, tz, k, axis_size=200):
    r_vect = np.asarray([rx, ry, rz])
    t_vect = np.asarray([tx, ty, tz])

    points = np.float32([[axis_size, 0, 0], [0, axis_size, 0], [0, 0, axis_size], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, r_vect, t_vect, k, (0, 0, 0, 0))

    img = copy.copy(image)

    cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
    cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
    cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

    return img


def rotate_image(image, rx, ry):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)

    rot_mat = cv2.getRotationMatrix2D(image_center, np.rad2deg(rx), 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    rot_mat = cv2.getRotationMatrix2D(image_center, np.rad2deg(ry), 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return image

