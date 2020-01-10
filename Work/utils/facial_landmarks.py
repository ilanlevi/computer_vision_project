import cv2

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

    return landmarks


def display_landmarks(img, shapes, name=''):
    # loop over the face detections
    for (i, shape) in enumerate(shapes):
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output - %s" % name, img)
