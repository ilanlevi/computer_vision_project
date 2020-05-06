import math

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
            cam_m, m = load_fpn_model(self.path_to_model, self.model_file_name, self.model_name)
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
        rx, ry, rz, tx, ty, tz = get_3d_pose(self.camera_matrix, self.model_matrix, landmarks)

        return rx, ry, rz, tx, ty, tz


# static methods
def save_pose(folder_path, img_name, pose):
    """
    Saves pose to '.pose' file
    :param folder_path: the folder path
    :param img_name: the image name
    :param pose: tuple or array of: (rx, ry, rz, tx, ty, tz)
    """
    np.savetxt(folder_path + img_name + '.pose', pose, fmt='%.4f', delimiter=', ',
               header='pitch, yaw, roll, tx, ty, tz')


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


def euler_2_rotation(euler_vector):
    """
    Change reparation of euler angle vector to face pose rotation vector
    :param euler_vector: reparation of euler angle vector (3, 1)
    :return: face pose rotation vector (3, 1)
    """
    euler_vector = np.squeeze(euler_vector)
    rotation = euler_vector_2_rotation_matrix(euler_vector)
    rotation_vector, _ = cv2.Rodrigues(rotation)
    rotation_vector = np.squeeze(rotation_vector)
    return rotation_vector


def euler_vector_2_rotation_matrix(euler_vector):
    """
    Euler angles vector to rotation matrices.
    Does that by calculating rotation matrix given euler angles.
    :param euler_vector: reparation of euler angle vector (3, 1)
    :return: face pose rotation matrix (3, 3)
    """

    # yaw
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(euler_vector[0]), -math.sin(euler_vector[0])],
                    [0, math.sin(euler_vector[0]), math.cos(euler_vector[0])]
                    ])
    # pitch
    R_y = np.array([[math.cos(euler_vector[1]), 0, math.sin(euler_vector[1])],
                    [0, 1, 0],
                    [-math.sin(euler_vector[1]), 0, math.cos(euler_vector[1])]
                    ])
    # roll
    R_z = np.array([[math.cos(euler_vector[2]), -math.sin(euler_vector[2]), 0],
                    [math.sin(euler_vector[2]), math.cos(euler_vector[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def rotation_2_euler(pose_vector):
    """
    Change reparation of face pose rotation vector to euler angle vector
    :param pose_vector: reparation of face pose rotation (3, 1)
    :return: euler angle vector (3, 1)
    """
    pose_matrix, _ = cv2.Rodrigues(pose_vector)
    pose_matrix = np.squeeze(pose_matrix)
    euler_vector = rotation_matrix_2_euler_vector(pose_matrix)
    return euler_vector


def rotation_matrix_2_euler_vector(rotation_matrix):
    """
    Rotation matrix to euler angles vector.
    Does that by calculating rotation matrix to euler angles
    :param rotation_matrix: face pose rotation matrix (3, 3)
    :return: euler angle vector (3, 1)
    """
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.array([x, y, z])
