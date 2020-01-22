import numpy as np

from abstract_read_data import AbstractReadData
from consts import DataSetConsts
from mytools import get_files_list
from utils import load_images, resize_image_and_landmarks
from mytools import get_pose


class ModelData(AbstractReadData):

    def __init__(self, data_path, image_size=None, picture_suffix=DataSetConsts.PICTURE_SUFFIX, to_gray=True,
                 to_hog=True, train_rate=DataSetConsts.DEFAULT_TRAIN_RATE, ):
        super(ModelData, self).__init__(data_path, image_size=image_size, train_rate=train_rate)

        self.data_path = data_path
        self.image_size = image_size
        self.original_file_list = []
        self.target_file_list = []
        self.picture_suffix = picture_suffix
        self.to_gray = to_gray
        self.to_hog = to_hog
        if not isinstance(picture_suffix, list):
            self.picture_suffix = [picture_suffix]

    def get_original_list(self):
        """
            return the dataset list of images from self.data_path +  self.original_sub
            :return self
        """
        files = get_files_list(self.data_path, self.picture_suffix)
        return files

    def pre_process_data(self):
        """
        Preprocess data (hog and size if defined)
        :return: self
        """
        if self.to_hog:



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

    def _init(self):
        self.original_file_list = self.get_original_list()
        self.read_data_set()
        self.pre_process_data()
        return self
