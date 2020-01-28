import cv2
import numpy as np

from consts import DataSetConsts
from image_utils import save_image
from mytools import count_files_in_dir, get_suffix, get_prefix, save_landmarks


# todo delte
class LandmarkWrapper:

    def __init__(self,
                 landmark_suffix=DataSetConsts.LANDMARKS_FILE_SUFFIX,
                 save_to_dir=None,
                 save_prefix=DataSetConsts.LANDMARKS_PREFIX,
                 ):
        """
        set to self values
        :param landmark_suffix: landmark file suffix (default is in DataSetConsts.LANDMARKS_FILE_SUFFIX)
        :param save_to_dir: what dir to save (None -> don't save)
        :param save_prefix: save image suffix
        """
        self.save_prefix = save_prefix
        self.save_to_dir = save_to_dir
        self.landmark_suffix = landmark_suffix

    def load_image_landmarks(self, image_path, new_image_shape=None):
        """
        :param image_path: full path to image
        :param new_image_shape: for rescaling - the original image size
        :exception ValueError: When cannot find landmarks file for image
        :return: image landmarks as np array
        """
        # landmarks = get_landmarks(image_path, self.landmark_suffix)

        prefix = get_prefix(image_path)
        path = prefix + self.landmark_suffix

        _, landmarks = cv2.face.loadFacePoints(path)
        if landmarks is None or []:
            raise ValueError("Cannot file landmarks for: " + image_path)
        landmarks = np.asarray(landmarks)
        landmarks = np.reshape(landmarks, (68, 2))
        if new_image_shape is not None:
            image = cv2.imread(image_path)
            original_shape = image.shape
            ratio_x = (new_image_shape[0] / float(original_shape[0]))
            ratio_y = (new_image_shape[1] / float(original_shape[1]))
            # resize landmarks
            landmarks = np.array(landmarks)
            landmarks[:, 0] = landmarks[:, 0] * ratio_y
            landmarks[:, 1] = landmarks[:, 1] * ratio_x

        return landmarks

    def _save_points_if_needed(self, image_path, landmark):
        """
        (This method will be used mainly is testing)
        Saves landmark image if needed (self.save_to_dir != None)
        :param image_path: the original_image_name
        :param landmark: the landmark array to save
        """
        if self.save_to_dir is None:
            return
        image_dir = get_prefix(image_path, '\\') + '\\'
        image_name = get_suffix(image_path, '\\')
        image_name = get_prefix(image_name)

        next_index = count_files_in_dir(image_dir, DataSetConsts.LANDMARKS_FILE_SUFFIX)
        landmark_image_name = image_dir + str(next_index) + '_' + image_name + DataSetConsts.LANDMARKS_FILE_SUFFIX
        save_landmarks(landmark_image_name, landmark)

    def _save_image_if_needed(self, image_path, landmark_image):
        """
        (This method will be used mainly is testing)
        Saves landmark image if needed (self.save_to_dir != None)
        :param image_path: the original_image_name
        :param landmark_image: the landmark image to save
        """
        if self.save_to_dir is None:
            return
        image_dir = get_prefix(image_path, '\\') + '\\'
        image_name = get_suffix(image_path, '\\')
        image_suffix = get_suffix(image_name)

        next_index = count_files_in_dir(image_dir, image_suffix)
        landmark_image_name = image_dir + str(next_index) + DataSetConsts.LANDMARKS_PREFIX + image_name
        save_image(landmark_image, landmark_image_name)
