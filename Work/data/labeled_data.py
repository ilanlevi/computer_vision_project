import numpy as np

from abstract_read_data import AbstractReadData
from ..consts import DataSetConsts
from ..mytools import get_files
from ..utils import load_images, get_landmarks


class LabeledData(AbstractReadData):

    def __init__(self, data_path, target_sub, label_file_name=None, random_state=DataSetConsts.DEFAULT_RANDOM_STATE,
                 train_rate=DataSetConsts.DEFAULT_TRAIN_RATE, image_size=DataSetConsts.PICTURE_WIDTH,
                 picture_suffix=DataSetConsts.PICTURE_SUFFIX, split_data=False, to_gray=True):
        super(LabeledData, self).__init__(data_path, random_state, train_rate, image_size)
        self.data_path = data_path
        self.label_file_name = label_file_name
        self.target_sub = target_sub
        self.random_state = random_state
        self.train_rate = train_rate
        self.image_size = image_size
        self.original_file_list = []
        self.target_file_list = []
        self.picture_suffix = picture_suffix
        self.split_data = split_data
        self.to_gray = to_gray

    def get_original_list(self):
        """
            return the dataset list of images from self.data_path +  self.original_sub
            :return self
        """
        files = get_files(self.data_path, self.picture_suffix)
        return files

    # abstracts:
    def read_data_set(self):
        tmp_x_train_set = load_images(self.original_file_list, gray=self.to_gray)
        self.x_train_set = []
        self.y_train_set = []
        for index in range(len(self.original_file_list)):
            ldmk_list = get_landmarks(self.original_file_list[index])
            if ldmk_list is not None:
                ldmk_list = np.asarray(ldmk_list)
                self.y_train_set.append(ldmk_list)
                self.x_train_set.append(tmp_x_train_set[index])

        self.y_train_set = np.asarray(self.y_train_set)
        self.x_train_set = np.asarray(self.x_train_set)

        return self

    def _init(self, unzip_file=True, path_to_use=None):
        self.original_file_list = self.get_original_list()
        return self
