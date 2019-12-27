import time

# import the necessary packages
import dlib
from collections import OrderedDict
import numpy as np
import cv2

from image_tools import ImageTools


class PreProcessDataExternal:
    FACIAL_LANDMARKS_IDXS = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17))
    ])

    FACIAL_LANDMARKS_5_IDXS = OrderedDict([
        ("right_eye", (2, 3)),
        ("left_eye", (0, 1)),
        ("nose", 4)
    ])

    def __init__(self, predictor_path=None):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def get_shapes(self, gray):
        results = []
        # detect faces in the grayscale image
        rects = self.detector(gray, 1)

        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = PreProcessDataExternal.rect_to_bb(rect)
            results.append((x, y, w, h))
        return results

    @staticmethod
    def draw_on_image(predictor, detector, image, gray):
        rects = detector(gray, 1)
        # loop over the face detections
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = PreProcessDataExternal.shape_to_np(shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = PreProcessDataExternal.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # show the face number
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output", image)
        cv2.waitKey(0)

    @staticmethod
    def rect_to_bb(rect):
        # take a bounding predicted by dlib and convert it
        # to the format (x, y, w, h) as we would normally do
        # with OpenCV
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        # return a tuple of (x, y, w, h)
        return x, y, w, h

    @staticmethod
    def shape_to_np(shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((shape.num_parts, 2), dtype=dtype)

        # loop over all facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, shape.num_parts):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords
