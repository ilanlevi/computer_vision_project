import cv2
import numpy as np

import scipy.io as scio


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
    # _, rotation_vec, translation_vec, _ = cv2.solvePnPRansac(model_matrix, landmarks, camera_matrix, None)
    _, rotation_vec, translation_vec = cv2.solvePnP(model_matrix, landmarks, camera_matrix, None)
    rotation_vec = np.squeeze(rotation_vec)
    translation_vec = np.squeeze(translation_vec)

    rx = rotation_vec[0]
    ry = rotation_vec[1]
    rz = rotation_vec[2]

    tx = translation_vec[0]
    ty = translation_vec[1]
    tz = translation_vec[2]

    return rx, ry, rz, tx, ty, tz

