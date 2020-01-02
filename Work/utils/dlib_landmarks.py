import cv2
import dlib
import numpy as np


class DlibLandmarks:

    def __init__(self, predictor_path=None):
        self.detector, self.predictor = DlibLandmarks.load_dlib(predictor_path)

    def get_landmarks(self, img):
        return DlibLandmarks._get_landmarks(img, self.detector, self.predictor)

    @staticmethod
    def _get_landmarks(img, detector, predictor):
        lmarks = []
        dets, scores, idx = detector.run(img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        shapes = []
        for k, det in enumerate(dets):
            shape = predictor(img, det)
            shapes.append(shape)
            xy = DlibLandmarks._shape_to_np(shape)
            lmarks.append(xy)

        lmarks = np.asarray(lmarks, dtype='float32')
        return lmarks

    @staticmethod
    def display_landmarks(img, shapes, name=''):
        # loop over the face detections
        for (i, shape) in enumerate(shapes):
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        # show the output image with the face detections + facial landmarks
        cv2.imshow("Output - %s" % name, img)

    @staticmethod
    def _shape_to_np(shape):
        xy = []
        for i in range(68):
            xy.append((shape.part(i).x, shape.part(i).y,))
        xy = np.asarray(xy, dtype='float32')
        return xy

    @staticmethod
    def load_dlib(predictor_path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        return detector, predictor
