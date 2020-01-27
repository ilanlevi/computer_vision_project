import numpy as np

from consts import DataSetConsts
from image_utils import save_image
from mytools import get_landmarks, count_files_in_dir, get_suffix, get_prefix, save_landmarks


class LandmarkWrapper:

    def __init__(self,
                 out_image_size=DataSetConsts.PICTURE_SIZE,
                 landmark_suffix=DataSetConsts.LANDMARKS_FILE_SUFFIX,
                 save_to_dir=None,
                 save_prefix=DataSetConsts.LANDMARKS_PREFIX
                 ):
        """
        set to self values
        :param landmark_suffix: landmark file suffix (default is in DataSetConsts.LANDMARKS_FILE_SUFFIX)
        :param save_to_dir: what dir to save (None -> don't save)
        :param save_prefix: save image suffix
        :param out_image_size: out image size (for rescale)
        """
        self.save_prefix = save_prefix
        self.save_to_dir = save_to_dir
        self.landmark_suffix = landmark_suffix
        self.out_image_size = out_image_size

    @staticmethod
    def apply_matrix_to_landmarks(matrix_to_apply, landmarks):
        new_image_map = np.zeros((68, 2), dtype=np.float)

        for i, landmark in enumerate(landmarks):
            curr_pixel = [landmark[0], landmark[1], 1]

            curr_pixel = np.dot(matrix_to_apply, curr_pixel)
            new_image_map[i][0] = curr_pixel[0]
            new_image_map[i][1] = curr_pixel[1]

        return new_image_map

    # todo - delete
    def get_transform_landmarks(self, path, landmarks_image):
        """
        :param path: the original image path
        :param landmarks_image: the landmark image
        :return: image landmarks as np array
        """
        landmarks_points = np.argwhere(landmarks_image != [0])
        # should be (68, 2)
        landmarks_points = np.reshape(landmarks_points, (68, 2))
        self._save_points_if_needed(path, landmarks_points)
        return landmarks_points

    def get_landmark_image(self, image_path, image_size=None):
        """
        creates the landmark image and saves if wished
        :param image_path: the image full path to get the landmark image
        :param image_size: the output size (image_size, image_size)
        :return: the landmark image (black image with landmarks in white)
        """
        landmarks = self.load_image_landmarks(image_path)
        landmarks_image = self.get_landmark_image_from_landmarks(landmarks, image_size)
        self._save_image_if_needed(image_path, landmarks_image)
        return landmarks_image

    def get_landmark_image_from_landmarks(self, landmarks, image_size=None):
        """
        creates the landmark image and saves if wished
        :param landmarks: the image landmarks
        :param image_size: the output size (image_size, image_size)
        :return: the landmark image (black image with landmarks in white)
        """
        if image_size is None:
            image_size = self.out_image_size
        landmarks_image = np.zeros((image_size, image_size))
        copy_l = landmarks.copy()
        copy_l = copy_l.astype(np.int)
        for point in copy_l:
            landmarks_image[point[1], point[0]] = 255
        return landmarks_image

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
