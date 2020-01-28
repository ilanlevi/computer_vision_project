from consts import FPNConsts
from models import get_3d_pose, load_fpn_model


class MyFpnWrapper:

    def __init__(self,
                 path_to_model=FPNConsts.LOCAL_PATH,
                 model_file_name=FPNConsts.POSE_P,
                 model_name=FPNConsts.MODEL_NAME):
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

    # @staticmethod
    # def apply_transformation_matrix_on_pose(pose_6DoF, matrix, flip_horizontal=False):
    #     if matrix is None:
    #         return pose_6DoF
    #
    #     rx, ry, rz, tx, ty, tz = pose_6DoF
    #     # matrix is in deg
    #     matrix = np.deg2rad(matrix)
    #     rotation_vector = np.asarray([rx, ry, rz])
    #     rotation_vector = np.reshape(rotation_vector, (3, 1))
    #     rotation_vector = matrix.dot(rotation_vector)
    #     rotation_vector = rotation_vector.flatten()
    #
    #     if flip_horizontal:
    #         # flip yaw (ry), roll (rz)
    #         rotation_vector[1] = -rotation_vector[1]
    #         rotation_vector[2] = -rotation_vector[2]
    #
    #     translation_vector = np.asarray([tx, ty, tz])
    #     translation_vector = np.reshape(translation_vector, (3, 1))
    #     translation_vector = matrix.dot(translation_vector)
    #     translation_vector = translation_vector.flatten()
    #
    #     new_pose = (rotation_vector[0],
    #                 rotation_vector[1],
    #                 rotation_vector[2],
    #                 translation_vector[0],
    #                 translation_vector[1],
    #                 translation_vector[2],
    #                 )
    #     return new_pose
