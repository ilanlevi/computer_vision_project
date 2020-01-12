import cv2
import numpy as np
import copy
from ..mytools import get_prefix


# todo comment
def get_landmarks(img_path, landmarks_suffix='.pts', print_data=False):
    prefix = get_prefix(img_path)
    path = prefix + landmarks_suffix
    # landmarks = []
    try:
        with open(path) as f:
            rows = [rows.strip() for rows in f]

        """Use the curly braces to find the start and end of the point data"""
        head = rows.index('{') + 1
        tail = rows.index('}')

        """Select the point data split into coordinates"""
        raw_points = rows[head:tail]
        coords_set = [point.split() for point in raw_points]

        """Convert entries from lists of strings to tuples of floats"""
        points = [tuple([float(point) for point in coords]) for coords in coords_set]
        return points
    #
    #     with open(path, mode='r') as f_marks:
    #         contents = f_marks.readlines()
    #         for i in range(2, len(contents) - 1):
    #             row = contents[i]
    #             pos = row.rfind(' ')
    #             if pos != -1:
    #                 try:
    #                     n1 = float(row[:pos])
    #                     n2 = float(row[pos + 1:])
    #                     landmarks.append((n1, n2))
    #                 except Exception as e:
    #                     if print_data:
    #                         print 'Nan: %s, Error: %s' % (row, str(e))
    except Exception as e:
        if print_data:
            print 'Error: ' + str(e)
    return None

    # return landmarks


def display_landmarks(img, landmarks, name=''):
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
    cv2.imshow("Output - %s" % name, img)


def display_bbox(img, lndmarks, name='', factor=0.10):
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
