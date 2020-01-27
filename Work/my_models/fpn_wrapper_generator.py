from consts import FPNConsts
from models import get_3d_pose, load_fpn_model


class FpnWrapper:

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

    def get_3d_vectors(self, landmarks, vertical_flip=False):
        """
        Uses get_3d_pose from fpn
        :param landmarks: image landmarks (68x2)
        :param vertical_flip: will be flipped or not (default is false)
        :return: rx, ry, rz, tx, ty, tz - face pose estimation
        """
        rx, ry, rz, tx, ty, tz = get_3d_pose(self.camera_matrix, self.model_matrix, landmarks)

        # flip ry, rz if vertical_flipped
        if vertical_flip:
            ry = -ry
            rz = -rz

        return rx, ry, rz, tx, ty, tz
