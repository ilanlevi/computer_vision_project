import time
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.model_selection import train_test_split

from consts.ds_consts import DataSetConsts


# todo - remove
class AbstractReadData:
    __metaclass__ = ABCMeta

    def __init__(self, data_path='', random_state=DataSetConsts.DEFAULT_RANDOM_STATE,
                 train_rate=DataSetConsts.DEFAULT_TRAIN_RATE, valid_rate=DataSetConsts.DEFAULT_VALID_RATE,
                 image_size=DataSetConsts.PICTURE_WIDTH):
        self.data_path = data_path
        self.image_size = image_size
        self.random_state = random_state
        self.train_rate = train_rate
        self.valid_rate = valid_rate
        self.x_train_set = self.y_train_set = \
            self.x_test_set = self.y_test_set = \
            self.x_valid_set = self.y_valid_set = []

    def get_picture_size(self):
        """
            :return image size
        """
        return self.image_size

    def get_picture_size_square(self):
        """
            :return (image size)^2
        """
        return self.image_size * self.image_size

    def split_dataset(self, data=None, labels=None, random_state=None, split_rate=None, valid_rate=None):
        """
            Uses sklearn.model_selection import train_test_split to split local data, set new values
            :return self
        """
        if data is None:
            data = self.x_train_set
        if labels is None:
            labels = self.y_train_set
        if split_rate is None:
            split_rate = self.train_rate
        if valid_rate is None:
            valid_rate = self.valid_rate
        data = np.asarray(data)
        labels = np.asarray(labels)
        self.x_train_set, self.x_test_set, self.y_train_set, self.y_test_set = train_test_split(
            data, labels, test_size=split_rate, random_state=random_state)
        self.x_test_set, self.x_valid_set, self.y_test_set, self.y_valid_set = train_test_split(
            self.x_test_set, self.y_test_set, test_size=valid_rate, random_state=random_state)
        return self

    def init(self, **kwargs):
        """
        measure time and call self.__init
        :param kwargs: the args for self._init
        :return: self
        """
        start = time.time()
        r = self._init(**kwargs)
        end = time.time()
        print('loading and init data took: %.2f sec.' % (end - start))
        return r

    def set_size(self, train_size, test_size):
        """
        Re-split train and test sets to get the required sizes
        :param train_size: the wanted size (int)
        :param test_size: the wanted size (int)
        :return: self
        """
        self.x_train_set, self.x_test_set, self.y_train_set, self.y_test_set = train_test_split(
            self.x_train_set, self.y_train_set, train_size=train_size, test_size=test_size,
            random_state=self.random_state)
        return self

    # abstracts:

    @abstractmethod
    def read_data_set(self):
        """
            Read file contents from file_name.
        """
        pass

    @abstractmethod
    def _init(self, **kwargs):
        """
            Initialize all
        """
        pass
