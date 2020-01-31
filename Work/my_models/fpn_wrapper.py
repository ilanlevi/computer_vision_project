import cv2
import numpy as np
import scipy.io as scio

from consts import FPN_LOCAL_PATH, POSE_P, FPN_MODEL_NAME


class FpnWrapper:

    def __init__(self,
                 path_to_model=FPN_LOCAL_PATH,
                 model_file_name=POSE_P,
                 model_name=FPN_MODEL_NAME):
        """
        Load and init the fpn model files
        :param path_to_model: directory to file
        :param model_file_name: model file name
        :param model_name: model name
        :exception ValueError: when model couldn't load
        """
        self.model_name = model_name
        self.model_file_name = model_file_name
        self.path_to_model = path_to_model
        try:
            cam_m, m = FpnWrapper.load_fpn_model(self.path_to_model, self.model_file_name, self.model_name)
        except Exception as e:
            error_msg = "Exception was raised while loading model! Error = %s" % str(e)
            raise ValueError(error_msg)

        self.camera_matrix = cam_m
        self.model_matrix = m

    def get_3d_vectors(self, landmarks):
        """
        Uses get_3d_pose from fpn
        :param landmarks: image landmarks (68x2)
        :return: rx, ry, rz, tx, ty, tz - face pose estimation
        """
        rx, ry, rz, tx, ty, tz = FpnWrapper.get_3d_pose(self.camera_matrix, self.model_matrix, landmarks)

        return rx, ry, rz, tx, ty, tz

    # static methods

    @staticmethod
    def save_pose(folder_path, img_name, pose):
        """
        Saves pose to '.pose' file
        :param folder_path: the folder path
        :param img_name: the image name
        :param pose: tuple or array of: (rx, ry, rz, tx, ty, tz)
        """
        np.savetxt(folder_path + img_name + '.pose', pose, fmt='%.4f', delimiter=', ',
                   header='pitch, yaw, roll, tx, ty, tz')

    @staticmethod
    def load_fpn_model(path_to_model, model_file_name, model_name):
        """
        Load FPN model from file.
            See: https://github.com/fengju514/Face-Pose-Net
        :param path_to_model: directory to file
        :param model_file_name: model file name
        :param model_name: model name
        :return: camera_matrix, model_matrix from file
        """
        path = path_to_model + model_file_name
        model = scio.loadmat(path)[model_name]
        camera_matrix = np.asmatrix(model['outA'][0, 0], dtype='float32')  # 3x3
        model_matrix = np.asarray(model['threedee'][0, 0], dtype='float32')  # 68x3
        return camera_matrix, model_matrix

    @staticmethod
    def get_3d_pose(camera_matrix, model_matrix, landmarks):
        """
        Calculate rotation and translation vectors using open-cv: solvePnP
        :param camera_matrix: the camera matrix from model
        :param model_matrix: the model matrix from model
        :param landmarks: image landmarks (68x2)
        :return: rx, ry, rz, tx, ty, tz - face pose estimation
        """
        _, rotation_vec, translation_vec = cv2.solvePnP(model_matrix, landmarks, camera_matrix, None)

        rotation_vec = np.squeeze(rotation_vec)
        translation_vec = np.squeeze(translation_vec)

        # set pitch, yaw, roll
        rx = rotation_vec[0]
        ry = rotation_vec[1]
        rz = rotation_vec[2]

        # set translation vector
        tx = translation_vec[0]
        ty = translation_vec[1]
        tz = translation_vec[2]

        return rx, ry, rz, tx, ty, tz
