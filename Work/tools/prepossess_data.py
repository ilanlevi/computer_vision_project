import time

# import the necessary packages
import dlib
from collections import OrderedDict
import numpy as np
import imutils
import cv2


class PreProcessData:

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    FACIAL_LANDMARKS_IDXS = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17))
    ])

    @staticmethod
    def rect_to_bb(rect):
        """
        take a bounding predicted by dlib and convert it
        to the format (x, y, w, h) as we would normally do with OpenCV
        :param rect: the rectangle
        :return: tuple of (x, y, w, h)
        """
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y

        return x, y, w, h

    @staticmethod
    def shape_to_np(shape, dtype="int"):
        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype=dtype)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return coords

    @staticmethod
    def convert_images(images, width=500):
        """
        for each image: resize image and convert to gray scale
        :param width: the width
        :param images: the loaded image
        :return: gray scaled image list
        """
        gray_images = []
        for im in images:
            gray = PreProcessData.convert_image(im, width)
            gray.append(gray)
        return gray_images

    @staticmethod
    def convert_image(image, width=500):
        """
        resize image and convert to gray scale
        :param width: the width
        :param image: the loaded image
        :return: gray scaled image
        """
        im = imutils.resize(image, width=width)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def detector(image):
        image = imutils.resize(image, width=500)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale image
        rects = detector(gray, 1)

    def _test(self, image_to_test):
        gray = PreProcessData.convert_image(image_to_test)
        # detect faces in the grayscale image
        rects = detector(gray, 1)

# def __init__(self, ):
