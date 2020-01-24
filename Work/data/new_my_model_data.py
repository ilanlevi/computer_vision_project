import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence

from consts import DataSetConsts
from image_utils import load_image, auto_canny
from mytools import get_files_list, get_pose


# todo - comments
class KerasModelData(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, data_path, dim=160, picture_suffix=DataSetConsts.PICTURE_SUFFIX, to_gray=True,
                 to_hog=True, sigma=0.33, batch_size=64, shuffle=True, to_fit=False, out_dim=6,
                 test_rate=DataSetConsts.DEFAULT_TEST_RATE, valid_rate=DataSetConsts.DEFAULT_VALID_RATE,
                 original_file_list=None):

        self.data_path = data_path
        self.dim = dim * dim
        self.picture_size = dim
        self.out_dim = out_dim
        self.picture_suffix = picture_suffix
        self.to_gray = to_gray
        self.to_hog = to_hog
        self.sigma = sigma
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.test_rate = test_rate
        self.valid_rate = valid_rate

        if not isinstance(picture_suffix, list):
            self.picture_suffix = [picture_suffix]
        if original_file_list is None:
            self.original_file_list = self._get_original_list()
        else:
            self.original_file_list = original_file_list
        self.indexes = np.arange(len(self.original_file_list))
        self.on_epoch_end()

    def _get_original_list(self):
        """
            return the dataset list of images from self.data_path +  self.original_sub
            :return the file list
        """
        files = get_files_list(self.data_path, self.picture_suffix)
        return np.asarray(files)

    def split_to_train_and_validation(self, t_size=None, v_size=None):
        """
        Split arrays or matrices into random train, test and validation subsets.
        Updates self.original_file_list as well (to test value)
        :param t_size: should be [0.0 ,1.0]. represent the proportion of the dataset to include in the test split split
        :param v_size: should be [0.0 ,1.0]. represent the proportion of the dataset to include in the validation split
        :return: tuple (test_files, validation_files).
        """
        if t_size is None:
            t_size = self.test_rate
        if v_size is None:
            v_size = self.valid_rate
        self.original_file_list, test_files = train_test_split(self.original_file_list, test_size=t_size,
                                                               shuffle=True)
        test_files, validation_files = train_test_split(test_files, test_size=v_size, shuffle=True)
        self.on_epoch_end()
        return test_files, validation_files

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.original_file_list) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.original_file_list[k] for k in indexes]

        # Generate data
        X = self._generate_X(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.original_file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self._load_image(ID)

        return X

    def _generate_y(self, list_IDs_temp):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, self.out_dim), dtype=np.float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i,] = np.asarray(get_pose(self.original_file_list[i]), dtype=np.float)

        return y

    def _load_image(self, image_path):
        """
        :param image_path: path to image to load
        :return: loaded image
        """
        tmp_image = load_image(image_path, size=self.picture_size, gray=self.to_gray)
        tmp_image = np.asarray(tmp_image, dtype=np.uint8)
        if self.to_hog:
            tmp_image = auto_canny(tmp_image, self.sigma)
        tmp_image = np.asarray(tmp_image, dtype=np.float)
        tmp_image = tmp_image.flatten()
        tmp_image = tmp_image / 255
        return tmp_image
