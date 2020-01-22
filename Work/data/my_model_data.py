import numpy as np
from sklearn.preprocessing import StandardScaler

from abstract_read_data import AbstractReadData
from consts import DataSetConsts
from mytools import get_files_list
from mytools import get_pose
from utils import load_images, auto_canny


class ModelData(AbstractReadData):

    def __init__(self, data_path, image_size=None, picture_suffix=DataSetConsts.PICTURE_SUFFIX, to_gray=True,
                 to_hog=True, train_rate=DataSetConsts.DEFAULT_TRAIN_RATE, sigma=0.33):
        super(ModelData, self).__init__(data_path, image_size=image_size, train_rate=train_rate)

        self.data_path = data_path
        self.image_size = image_size
        self.original_file_list = []
        self.target_file_list = []
        self.picture_suffix = picture_suffix
        self.to_gray = to_gray
        self.to_hog = to_hog
        self.sigma = sigma
        if not isinstance(picture_suffix, list):
            self.picture_suffix = [picture_suffix]

    def get_original_list(self):
        """
            return the dataset list of images from self.data_path +  self.original_sub
            :return self
        """
        files = get_files_list(self.data_path, self.picture_suffix)
        return files

    def canny_filter(self, sigma=None):
        """
        Apply canny edge detection on both train and test set
        :param sigma: sigma threshold factor - default is self
        :return: self
        """
        if sigma is None:
            sigma = self.sigma
        for index in range(len(self.x_test_set)):
            self.x_test_set[index] = auto_canny(self.x_test_set[index], sigma)
        for index in range(len(self.x_train_set)):
            self.x_train_set[index] = auto_canny(self.x_train_set[index], sigma)

        return self

    def normalize_data(self):
        # normalize the data
        std = StandardScaler()
        self.x_train_set = std.fit(self.x_train_set)
        self.x_test_set = std.fit(self.x_test_set)

    # abstracts:
    def read_data_set(self):
        tmp_x_train_set = load_images(self.original_file_list, gray=self.to_gray)
        self.x_train_set = []
        self.y_train_set = []
        for index in range(len(self.original_file_list)):
            img_pose = get_pose(self.original_file_list[index])
            if img_pose is not None:
                self.y_train_set.append(img_pose)
                self.x_train_set.append(tmp_x_train_set[index])
            else:
                print '> Cannot find pose, ignoring image: ' + self.original_file_list[index]

        self.y_train_set = np.asarray(self.y_train_set)
        self.x_train_set = np.asarray(self.x_train_set)

        return self

    def _init(self, preprocess=False):
        self.original_file_list = self.get_original_list()
        self.read_data_set()
        return self
