import gzip
from trace import pickle
import numpy as np

from abstract_read_data import AbstractReadData


class MNIST(AbstractReadData):
    # const
    FILE_NAME = 'mnist.gz'
    DEFAULT_TRAIN_RATE = 1 / 7.0
    PICTURE_SIZE = 28

    NUMBER_OF_DIGITS = 10
    NUMBER_OF_DIGITS_TO_SHOW = 12

    def __init__(self, data_path='', file_name=FILE_NAME):
        super(MNIST, self).__init__(data_path)
        self.image_size = MNIST.PICTURE_SIZE
        self.train_rate = MNIST.DEFAULT_TRAIN_RATE
        self.file_name = file_name

    def count_digit(self, check_test_set=True):
        """
            count how many times each digit appears
            :return 10 sized array
        """
        count_arr = np.zeros(10)
        for label in self.y_train_set[:]:
            count_arr[label] += 1
        if check_test_set:
            for label in self.y_test_set[:]:
                count_arr[label] += 1
        return count_arr

    # abstracts:
    def read_data_set(self):
        """
        use mnist archive to read data
        :return:
        """
        p = self.data_path + self.file_name
        try:
            with gzip.open(p, 'rb') as f:
                u = pickle.Unpickler(f)
                u.encoding = 'latin1'
                return u.load()
        except Exception as e:
            print 'Error while extracting file! ' + str(e)
            # error while reading db, try without the encoding
            with gzip.open(p, 'rb') as f:
                u = pickle.Unpickler(f)
                return u.load()

    def read_data_set_and_separate(self):
        train, self.valid_set, test = self.read_data_set()
        self.x_train_set, self.y_train_set = train
        self.x_test_set, self.y_test_set = test
        return self

    def _init(self, **kwargs):
        return self.read_data_set_and_separate().split_dataset()
