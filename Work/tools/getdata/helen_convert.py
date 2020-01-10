import os
import numpy as np

from abstract_create_data import AbstractConvertData


class HelenDataSet(AbstractConvertData):
    # def __init__(self, data_path='', output_file='', print_data=False):
    def __init__(self, data_path, output_file_name, print_data=None, random_state=DataSetConsts.DEFAULT_RANDOM_STATE,
                 train_rate=DataSetConsts.DEFAULT_TRAIN_RATE, image_size=DataSetConsts.PICTURE_WIDTH,
                 picture_suffix=DataSetConsts.PICTURE_SUFFIX, split_data=False, to_gray=True):
        super(HelenDataSet, self).__init__(data_path, random_state, train_rate, image_size)
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

        files = [os.path.join(r, file_) for r, d, f in os.walk(self.data_path) for file_ in f]
        f_list = []
        for file_name in files:
            if self.picture_suffix in file_name:
                f_list.append(file_name)
        return f_list

    @staticmethod
    def get_file_type(file_name):
        """
            Split file label from the name (data set structure)
            :return the file label
        """
        split, _ = file_name.split('_', 1)
        return split

    # abstracts:
    def read_data_set(self):
        if self.to_gray:
            self.x_train_set = imT.load_converted_images(self.original_file_list)
        else:
            self.x_train_set = imT.load_images(self.original_file_list)

        self.y_train_set = []
        if self.label_file_name is not None:
            labels = read_csv(self.data_path, self.label_file_name)
            if labels:
                self.y_train_set = np.asarray(labels)

        return self

    def _init(self, unzip_file=True, path_to_use=None):
        self.original_file_list = self.get_original_list()
        return self
