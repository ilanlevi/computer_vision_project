import cv2
import numpy as np

from ..mytools import get_prefix


# todo comment
def get_landmarks(img_path, landmarks_suffix='.pts', print_data=False):
    prefix = get_prefix(img_path)
    path = prefix + landmarks_suffix
    landmarks = []
    try:
        with open(path, mode='r') as f_marks:
            contents = f_marks.readlines()
            for i in range(2, len(contents) - 1):
                row = contents[i]
                pos = row.rfind(' ')
                if pos != -1:
                    try:
                        n1 = float(row[:pos])
                        n2 = float(row[pos + 1:])
                        landmarks.append((n1, n2))
                    except Exception as e:
                        if print_data:
                            print 'Nan: %s, Error: %s' % (row, str(e))
    except Exception as e:
        if print_data:
            print 'Error: ' + str(e)
        return None

    return landmarks


def display_landmarks(img, lndmarks, name=''):
    # loop over the face detections
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
