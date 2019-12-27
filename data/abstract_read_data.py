import time
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.model_selection import train_test_split


class AbstractReadData:
    __metaclass__ = ABCMeta

    # const
    DEFAULT_RANDOM_STATE = 0
    DEFAULT_TRAIN_RATE = 1 / 7.0

    def __init__(self, data_path=''):
        self.file_name = data_path
        self.data_path = data_path
        self.image_size = 0
        self.random_state = AbstractReadData.DEFAULT_RANDOM_STATE
        self.train_rate = AbstractReadData.DEFAULT_TRAIN_RATE
        self.x_train_set = self.y_train_set = self.x_test_set = self.y_test_set = self.valid_set = []

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

    def split_dataset(self, data=None, labels=None, random_state=None, split_rate=None):
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
        data = np.asarray(data)
        labels = np.asarray(labels)
        self.x_train_set, self.x_test_set, self.y_train_set, self.y_test_set = train_test_split(
            data, labels, test_size=split_rate, random_state=random_state)
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
        print 'loading and init data took: %.2f sec.' % (end - start)
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
    def read_data_set_and_separate(self):
        """"
            Read file contents from file_name.
            :returns and set to self the data as tuple
        """
        pass

    @abstractmethod
    def _init(self, **kwargs):
        """
            Initialize all
        """
        pass
