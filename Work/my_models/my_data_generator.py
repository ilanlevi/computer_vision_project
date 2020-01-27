import numpy as np
from keras_preprocessing.image import NumpyArrayIterator
from sklearn.model_selection import train_test_split

from consts import DataSetConsts
from image_utils import load_image, auto_canny
from mytools import get_files_list, get_pose


class MyDataGenerator(NumpyArrayIterator):

    def __init__(self,
                 data_path,
                 image_data_generator,
                 picture_suffix=DataSetConsts.PICTURE_SUFFIX,
                 image_landmarks_generator=None,
                 image_6DoF_generator=None,
                 original_file_list=None,
                 shuffle=True,
                 test_rate=DataSetConsts.DEFAULT_TEST_RATE,
                 valid_rate=DataSetConsts.DEFAULT_VALID_RATE,
                 sample_weight=None,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png'):

        self.save_format = save_format
        self.save_prefix = save_prefix
        self.save_to_dir = save_to_dir
        self.data_format = data_format
        self.seed = seed
        self.sample_weight = sample_weight
        self.valid_rate = valid_rate
        self.test_rate = test_rate
        self.shuffle = shuffle
        self.image_data_generator = image_data_generator
        self.image_landmarks_generator = image_landmarks_generator
        self.image_6DoF_generator = image_6DoF_generator
        self.data_path = data_path

        if not isinstance(picture_suffix, list):
            self.picture_suffix = [picture_suffix]
        if original_file_list is None:
            self.n = self._get_original_list()
        else:
            self.n = original_file_list

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
        self.n, test_files = train_test_split(self.n, test_size=t_size, shuffle=self.shuffle)
        test_files, validation_files = train_test_split(test_files, test_size=v_size, shuffle=self.shuffle)
        self.on_epoch_end()
        return test_files, validation_files

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
        X, delete_indexes = self._generate_x(list_IDs_temp)

        if self._to_fit:
            y = self._generate_y(list_IDs_temp, delete_indexes)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.original_file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_x(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X = np.empty((self.batch_size, 1, self.picture_size, self.picture_size))

        # Generate data
        delete_indexes = []
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            try:
                x = self._load_image(ID)
                X[i, 0,] = x
            except Exception as e:
                delete_indexes.append(i)
                print("_load_image for %s, error: %s" % (ID, str(e)))

        for index in delete_indexes:
            X = np.delete(X, index)

        return X, delete_indexes

    def _generate_y(self, list_IDs_temp, delete_indexes=None):
        """Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        """
        y = np.empty((self.batch_size, 1, self.out_dim), dtype=np.float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            y[i, 0,] = np.asarray(get_pose(ID), dtype=np.float)

        for index in delete_indexes:
            y = np.delete(y, index)

        return y

    def _load_image(self, image_path):
        """
        :param image_path: path to image to load
        :return: loaded image
        """

        tmp_image = load_image(image_path, size=self.picture_size, gray=self.to_gray)
        tmp_image = np.asarray(tmp_image, dtype=np.uint8)
        if self.do_canny:
            tmp_image = auto_canny(tmp_image, self.sigma)
        tmp_image = np.asarray(tmp_image, dtype=np.float)
        # todo delete
        # tmp_image = tmp_image.flatten()
        tmp_image = tmp_image / 255
        return tmp_image
