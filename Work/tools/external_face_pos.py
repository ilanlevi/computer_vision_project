import dlib
from collections import OrderedDict
import numpy as np
import cv2


class PreProcessDataExternal:
    FACIAL_LANDMARKS_68_IDXS = OrderedDict([
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

    def __init__(self, predictor_path=None, desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.desired_left_eye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

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

    def align(self, image, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = PreProcessDataExternal.shape_to_np(shape)

        if len(shape) == 68:
            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = PreProcessDataExternal.FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = PreProcessDataExternal.FACIAL_LANDMARKS_68_IDXS["right_eye"]
        else:
            (lStart, lEnd) = PreProcessDataExternal.FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = PreProcessDataExternal.FACIAL_LANDMARKS_5_IDXS["right_eye"]

        left_eye_pts = shape[lStart:lEnd]
        right_eye_pts = shape[rStart:rEnd]

        # compute the center of mass for each eye
        left_eye_center = left_eye_pts.mean(axis=0).astype("int")
        right_eye_center = right_eye_pts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        d_y = right_eye_center[1] - left_eye_center[1]
        d_x = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(d_y, d_x)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((d_x ** 2) + (d_y ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
        desired_dist *= self.desiredFaceWidth
        scale = desired_dist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        m = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        t_x = self.desiredFaceWidth * 0.5
        t_y = self.desiredFaceHeight * self.desired_left_eye[1]
        m[0, 2] += (t_x - eyes_center[0])
        m[1, 2] += (t_y - eyes_center[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, m, (w, h),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

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
