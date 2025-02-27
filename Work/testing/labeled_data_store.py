import numpy as np

from consts import PICTURE_SUFFIX
from image_utils import load_images, resize_image_and_landmarks, load_image_landmarks
from my_utils import get_files_list


class LabeledDataStore:

    def __init__(self, data_path, original_file_list=None, image_size=None, picture_suffix=PICTURE_SUFFIX,
                 to_gray=True):
        self.data_path = data_path
        self.image_size = image_size
        self.original_file_list = []
        self.target_file_list = []
        self.picture_suffix = picture_suffix
        self.to_gray = to_gray
        if not isinstance(picture_suffix, list):
            self.picture_suffix = [picture_suffix]
        if original_file_list is None:
            self.original_file_list = self.get_original_list()
        self.original_file_list = original_file_list
        self.y = []
        self.x = []

    def get_original_list(self):
        """
            return the dataset list of images from self.data_path +  self.original_sub
            :return self
        """
        files = get_files_list(self.data_path, self.picture_suffix)
        return files

    def read_data_set(self):
        self.y = []
        self.x = []
        new_file_list = []

        tmp_x_train_set = load_images(self.original_file_list, gray=self.to_gray)
        for index in range(len(tmp_x_train_set)):
            try:
                ldmk_list = load_image_landmarks(self.original_file_list[index])
                if ldmk_list is not None:
                    ldmk_list = np.asarray(ldmk_list)
                    im, lmarlk = resize_image_and_landmarks(tmp_x_train_set[index], ldmk_list, self.image_size)
                    self.y.append(lmarlk)
                    self.x.append(im)
                    new_file_list.append(self.original_file_list[index])
            except Exception:
                print("Ignoring file: " + self.original_file_list[index])

        self.y = np.asarray(self.y)
        self.x = np.asarray(self.x)
        self.original_file_list = np.asarray(new_file_list)

        return self
