from random import randint

import dlib
import numpy as np
import cv2

from Work.utils.my_io import _get_suffix
from Work.data.helen_data import HelenDataSet
from Work.utils.image_tools import ImageTools
from Work.consts.facial_landmarks import FacialLandmarksConsts as flc
from Work.consts.files_consts import HelenFileConsts as hfc


class CVPreProcess:
    # 3D model points.
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (0, -150.0, -125.0),  # Mouth
    ], dtype="float")

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

    def faces_pose(self, gray):
        output = []
        rects = self.detector(gray, 1)
        for rect in rects:
            o = self.calib_camera_image(gray, rect)
            output.append(o)

        return output

    def get_shapes(self, gray):
        results = []
        # detect faces in the grayscale image
        rect = self.detector(gray, 1)

        for (i, rect) in enumerate(rect):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            (x, y, w, h) = CVPreProcess.rect_to_bb(rect)
            results.append((x, y, w, h))

        return results

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

    def calib_camera_image(self, gray, rect):
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = self.predictor(gray, rect)
        shape = CVPreProcess.shape_to_np(shape)

        if len(shape) == 68:
            # extract the left and right eye (x, y)-coordinates
            (nStart, nEnd) = flc.FACIAL_LANDMARKS_68_INDEXES["nose"]
            (lStart, lEnd) = flc.FACIAL_LANDMARKS_68_INDEXES["left_eye"]
            (rStart, rEnd) = flc.FACIAL_LANDMARKS_68_INDEXES["right_eye"]
            (cStart, cEnd) = flc.FACIAL_LANDMARKS_68_INDEXES["chin"]
            (mStart, mEnd) = flc.FACIAL_LANDMARKS_68_INDEXES["mouth"]
        else:
            (nStart, nEnd) = flc.FACIAL_LANDMARKS_5_INDEXES["nose"]
            (lStart, lEnd) = flc.FACIAL_LANDMARKS_5_INDEXES["left_eye"]
            (rStart, rEnd) = flc.FACIAL_LANDMARKS_5_INDEXES["right_eye"]
            (cStart, cEnd) = (0, 0)
            (mStart, mEnd) = (0, 0)

        nose_pts = shape[nStart:nEnd]
        left_eye_pts = shape[lStart:lEnd]
        right_eye_pts = shape[rStart:rEnd]
        chin_pts = shape[cStart:cEnd]
        mouth_pts = shape[mStart:mEnd]

        # compute the center of mass for each eye
        # nose_center = nose_pts.mean(axis=0).astype("float")
        # left_eye_center = left_eye_pts.mean(axis=0).astype("float")
        # right_eye_center = right_eye_pts.mean(axis=0).astype("float")
        nose_center = np.asarray(nose_pts.mean(axis=0), dtype=np.float)
        left_eye_center = np.asarray(left_eye_pts.mean(axis=0), dtype=np.float)
        right_eye_center = np.asarray(right_eye_pts.mean(axis=0), dtype=np.float)
        chin_center = np.asarray(chin_pts.mean(axis=0), dtype=np.float)
        mouth_center = np.asarray(mouth_pts.mean(axis=0), dtype=np.float)

        eye_nose_center = np.array([nose_center, chin_center, left_eye_center, right_eye_center, mouth_center])

        size = gray.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)

        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float
        )

        print "Camera Matrix :\n " + str(camera_matrix)

        dist_coeffs = np.zeros((5, 1), dtype=np.float)  # Assuming no lens distortion

        (success, rotation_vector, translation_vector) = cv2.solvePnP(CVPreProcess.MODEL_POINTS,
                                                                      eye_nose_center,
                                                                      camera_matrix, dist_coeffs,
                                                                      flags=cv2.CV_ITERATIVE)

        # print "Rotation Vector:\n {0}".format(rotation_vector)
        print "Rotation Vector:\n" + str(rotation_vector)
        print "Translation Vector:\n" + str(translation_vector)

        return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs, eye_nose_center

    @staticmethod
    def draw_on_image(im, pose, name=''):
        success, rotation_vector, translation_vector, camera_matrix, dist_coeffs, image_points = pose
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector,
                                                         camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(im, p1, p2, (255, 0, 0), 2)

        # Display image
        cv2.imshow("Output - " + name, im)
        cv2.waitKey(0)

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


if __name__ == '__main__':
    NUMBER_OF_TESTS = 1

    ds = HelenDataSet(data_path=hfc.DOWNLOAD_FOLDER2, original_sub=hfc.VALID_SET_SUB_FOLDER,
                      target_sub=hfc.PROCESSED_SET_FOLDER, picture_suffix='png')
    ds.init()
    original_images = ds.original_file_list

    x = CVPreProcess(predictor_path=(hfc.DOWNLOAD_FOLDER + hfc.PREDICTOR_FILE_NAME))

    for i in range(NUMBER_OF_TESTS):
        q = 0
        rnd_index = randint(0, len(original_images) - 1)
        con_images = ImageTools.load_converted_images([ds.original_file_list[rnd_index]], 500)
        i_name = 'Image = ' + ds.original_file_list[rnd_index]
        print i_name
        faces_poses = x.faces_pose(con_images[0])
        for pose in faces_poses:
            p_name = _get_suffix((i_name + str(q)), '\\')
            print p_name
            x.draw_on_image(con_images[0], pose, p_name)
            q = q + 1
