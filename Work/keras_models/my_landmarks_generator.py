import numpy as np

from consts import DataSetConsts
from mytools import get_landmarks


class LandmarkWrapper:

    def __init__(self,
                 out_image_size=DataSetConsts.PICTURE_SIZE,
                 landmark_suffix=DataSetConsts.LANDMARKS_FILE_SUFFIX,
                 save_to_dir=None,
                 save_prefix=DataSetConsts.LANDMARKS_PREFIX,
                 save_format='png'
                 ):
        """
        set to self values
        :param landmark_suffix: landmark file suffix (default is in DataSetConsts.LANDMARKS_FILE_SUFFIX)
        :param save_to_dir: what dir to save (None -> don't save)
        :param save_prefix: save image suffix
        :param save_format: save format suffix
        :param out_image_size: out image size (for rescale)
        """
        self.save_format = save_format
        self.save_prefix = save_prefix
        self.save_to_dir = save_to_dir
        self.landmark_suffix = landmark_suffix
        self.out_image_size = out_image_size

    def transform_landmarks(self, image_path, ):
        """
        :param image_path: full path to image
        :exception ValueError: When cannot find landmarks file for image
        :return: image landmarks as np array
        """
        landmarks = get_landmarks(image_path, self.landmark_suffix)
        if landmarks is None or []:
            raise ValueError("Cannot file landmarks for: " + image_path)
        landmarks = np.asarray(landmarks)
        return landmarks

    def get_landmark_image(self, image_path, image_size=None):
        if image_size is None:
            image_size = self.out_image_size
        landmarks_image = np.zeros(())
        landmarks = self.load_image_landmarks(image_path)

    def load_image_landmarks(self, image_path):
        """
        :param image_path: full path to image
        :exception ValueError: When cannot find landmarks file for image
        :return: image landmarks as np array
        """
        landmarks = get_landmarks(image_path, self.landmark_suffix)
        if landmarks is None or []:
            raise ValueError("Cannot file landmarks for: " + image_path)
        landmarks = np.asarray(landmarks)
        return landmarks
