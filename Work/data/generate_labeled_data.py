import numpy as np

from consts import CsvConsts
from consts import DataSetConsts
from image_utils import load_images
from mytools import read_csv, get_files_list
from .abstract_read_data import AbstractReadData


# todo - remove
class GenerateLabeledData(AbstractReadData):
    """
    This class will be used for generating new labels for images
    """

    def __init__(self, data_path, image_size,
                 random_state=DataSetConsts.DEFAULT_RANDOM_STATE,
                 train_rate=DataSetConsts.DEFAULT_TRAIN_RATE,
                 to_gray=True, ):

        super(GenerateLabeledData, self).__init__(data_path, random_state, train_rate, image_size)

        self.data_path = data_path

        self.image_size = image_size
        self.original_file_list = []
        self.target_file_list = []
        self.to_gray = to_gray

    # abstracts:
    def read_data_set(self):
        self.original_file_list = []
        self.x_train_set = []
        self.y_train_set = []

        files = get_files_list(self.data_path, '.csv')

        for csv_file_path in files:
            csv_data = read_csv(csv_file_path)
            # convert each row
            for row in csv_data:
                label_value = dict(row)
                for float_value in CsvConsts.CSV_VALUES_LABELS:
                    label_value[float_value] = float(label_value[float_value])

                self.y_train_set.append(label_value)
                self.original_file_list.append(self.data_path + label_value[CsvConsts.PICTURE_NAME])

        self.y_train_set = np.asarray(self.y_train_set)
        self.original_file_list = np.asarray(self.original_file_list)

        self.x_train_set = load_images(self.original_file_list, gray=self.to_gray)
        self.x_train_set = np.asarray(self.x_train_set)

        return self

    def _init(self):
        self.read_data_set()
        return self
