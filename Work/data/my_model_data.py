import numpy as np
from sklearn.preprocessing import StandardScaler

from .abstract_read_data import AbstractReadData
from consts import DataSetConsts
from mytools import get_files_list
from mytools import get_pose
from image_utils import load_images, auto_canny


class ModelData(AbstractReadData):

    def __init__(self, data_path, image_size=160, picture_suffix=DataSetConsts.PICTURE_SUFFIX, to_gray=True,
                 to_hog=True, train_rate=DataSetConsts.DEFAULT_TRAIN_RATE, sigma=0.33, batch_size=500):
        super(ModelData, self).__init__(data_path, image_size=image_size, train_rate=train_rate)

        self.data_path = data_path
        self.image_size = image_size
        self.original_file_list = []
        self.target_file_list = []
        self.picture_suffix = picture_suffix
        self.to_gray = to_gray
        self.to_hog = to_hog
        self.sigma = sigma
        self.batch_size = batch_size
        self.read_index = 0
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
            self.x_test_set[index] = auto_canny(np.asarray(self.x_test_set[index], dtype=np.uint8), sigma)
        for index in range(len(self.x_train_set)):
            self.x_train_set[index] = auto_canny(np.asarray(self.x_train_set[index], dtype=np.uint8), sigma)
        for index in range(len(self.x_valid_set)):
            self.x_valid_set[index] = auto_canny(np.asarray(self.x_valid_set[index], dtype=np.uint8), sigma)

        self.x_train_set = self.my_flatten(self.x_train_set)
        if self.x_test_set:
            self.x_test_set = self.my_flatten(self.x_test_set)
        if self.x_valid_set:
            self.x_valid_set = self.my_flatten(self.x_valid_set)

        return self

    def normalize_data(self):
        # normalize the data
        scaler = StandardScaler()

        self.x_train_set = np.asarray(self.x_train_set)

        self.x_train_set = scaler.fit_transform(self.x_train_set.reshape(-1, self.x_train_set.shape[-1])) \
            .reshape(self.x_train_set.shape)
        if self.x_test_set:
            self.x_test_set = scaler.transform(self.x_test_set.reshape(-1, self.x_test_set.shape[-1])) \
                .reshape(self.x_test_set.shape)
        if self.x_valid_set:
            self.x_valid_set = scaler.transform(self.x_valid_set.reshape(-1, self.x_valid_set.shape[-1])) \
                .reshape(self.x_valid_set.shape)



    def my_flatten(self, threeD):
        shape = threeD.shape
        return np.reshape(threeD, (shape[0], shape[1] * shape[2]))

    # abstracts:
    def read_data_set(self):

        tmp_file_list = self.original_file_list[
                        self.read_index:min(self.read_index + self.batch_size, len(self.original_file_list) - 1)]
        self.read_index = min(self.read_index + self.batch_size, len(self.original_file_list) - 1)

        tmp_x_train_set = load_images(tmp_file_list, gray=self.to_gray)
        self.x_train_set = []
        self.y_train_set = []
        for index in range(len(tmp_x_train_set)):
            img_pose = get_pose(self.original_file_list[index + self.read_index])
            if img_pose is not None:
                self.y_train_set.append(img_pose)
                self.x_train_set.append(tmp_x_train_set[index])
            else:
                print('> Cannot find pose, ignoring image: ' + self.original_file_list[index])

        self.y_train_set = np.asarray(self.y_train_set)
        self.x_train_set = np.asarray(self.x_train_set)

        return self.read_index < (len(self.original_file_list) - 1)

    def _init(self, preprocess=False):
        self.original_file_list = self.get_original_list()
        return self
