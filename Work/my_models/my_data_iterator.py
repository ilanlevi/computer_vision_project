import numpy as np
from keras_preprocessing.image import Iterator
from sklearn.model_selection import train_test_split

from consts import DataSetConsts
from image_utils import load_image, auto_canny, resize_image_and_landmarks
from my_models import LandmarkWrapper, FpnWrapper
from mytools import get_files_list


class MyDataIterator(Iterator):

    def __init__(self,
                 data_path,
                 image_data_generator,
                 original_file_list=None,
                 batch_size=DataSetConsts.BATCH_SIZE,
                 picture_suffix=DataSetConsts.PICTURE_SUFFIX,
                 image_size=DataSetConsts.PICTURE_SIZE,
                 gen_y=False,
                 landmark_wrapper=LandmarkWrapper(),
                 fpn_wrapper=FpnWrapper(),
                 shuffle=False,
                 sample_weight=None,
                 seed=1,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 dtype='float32'):

        self.image_data_generator = image_data_generator
        self.data_path = data_path
        self.batch_size = batch_size
        self.gen_y = gen_y
        self.landmark_wrapper = landmark_wrapper
        self.fpn_wrapper = fpn_wrapper
        self.shuffle = shuffle
        self.sample_weight = sample_weight
        self.seed = seed
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.dtype = dtype
        self.image_size = image_size

        # set grayscale channel
        if self.data_format == 'channels_last':
            self.image_shape = (self.image_size, self.image_size) + (1,)
            self.landmark_shape = DataSetConsts.LANDMARKS_SHAPE + (1,)

        else:
            self.image_shape = (1,) + (self.image_size, self.image_size)
            self.landmark_shape = (1,) + DataSetConsts.LANDMARKS_SHAPE

        if not isinstance(picture_suffix, list):
            self.picture_suffix = [picture_suffix]
        if original_file_list is None:
            original_file_list = self._get_original_list()

        self.list_of_files = original_file_list

        super(MyDataIterator, self).__init__(len(self.list_of_files),
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             seed=seed)

    def _get_original_list(self):
        """
            return the dataset list of images from self.data_path +  self.original_sub
            :return the file list
        """
        files = get_files_list(self.data_path, self.picture_suffix)
        return np.asarray(files)

    def split_to_train_and_validation(self, t_size, v_size=None):
        """
        Split arrays or matrices into random train, test and validation subsets.
        Updates self.original_file_list as well (to test value)
        :param t_size: should be [0.0 ,1.0]. represent the proportion of the dataset to include in the test split split
        :param v_size: should be [0.0 ,1.0]. represent the proportion of the dataset to include in the validation split
        :return: tuple (test_files, validation_files).
        """
        self.list_of_files, test_files = train_test_split(self.list_of_files, test_size=t_size, shuffle=self.shuffle)
        if v_size is not None:
            test_files, validation_files = train_test_split(test_files, test_size=v_size, shuffle=self.shuffle)
        else:
            validation_files = []

        self.on_epoch_end()
        return test_files, validation_files

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(tuple([len(index_array)] + list(self.image_shape)[1:]), dtype=self.dtype)
        batch_y = np.zeros((len(index_array), 6), dtype=self.dtype)

        for i, j in enumerate(index_array):
            x, landmarks, y = self._get_transformed_samples(j)
            x_params = self.image_data_generator.get_random_transform(x.shape, seed=self.seed)
            afflined_keypoints = np.zeros(landmarks.shape)
            landmarks =, y = self._get_transformed_samples(j)
            x = self.image_data_generator.apply_transform(x.astype(self.dtype), x_params)

            # todo - check out
            # https://stackoverflow.com/questions/47970525/apply-an-affine-transform-to-a-bounding-rectangle
            #  https://stackoverflow.com/questions/18637494/camera-position-in-world-coordinate-from-cvsolvepnp
            # https://stackoverflow.com/questions/26500238/opencv-python-applying-rotation-matrix-from-rodrigues-function
            batch_x[i] = x

        if self.save_to_dir:
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=np.random.randint(1e4),
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        batch_x_miscs = [xx[index_array] for xx in self.x_misc]
        output = (batch_x if batch_x_miscs == []
                  else [batch_x] + batch_x_miscs,)
        if self.y is None:
            return output[0]
        output += (self.y[index_array],)
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output

    def _get_transformed_samples(self, index):
        """
        :param index: index from self.list_of_files
        :return: tuple of: (loaded image after canny, landmarks or None if !self.gen_y, 6DoF or None if !self.gen_y)
        """
        image, landmarks = self._load_image(self.list_of_files[index])
        y = None
        if self.gen_y:
            y = self.fpn_wrapper.get_3d_vectors(landmarks)
            y = np.asarray(y, dtype=np.float)
        return image, landmarks, y

    def _transform_all(self, image, landmarks, pose):
        image = np.reshape(image, self.image_shape)
        landmarks = np.reshape(landmarks, self.landmark_shape)

        image_param = self.image_data_generator.get_random_transform(image.shape, seed=self.seed)

    def _load_image(self, image_path):
        """
        :param image_path: path to image to load
        :return: tuple of: (loaded image after canny, landmarks or None if !self.gen_y)
        """
        if self.gen_y:
            landmarks = self.landmark_wrapper.load_image_landmarks(image_path)
            image = load_image(image_path)
            image, landmarks = resize_image_and_landmarks(image, landmarks, self.image_size)
            self.landmark_wrapper.get_transform_landmarks(image_path, landmarks)
        else:
            image = load_image(image_path, self.image_size)
            landmarks = None

        image = np.asarray(image, dtype=np.uint8)
        image = auto_canny(image, DataSetConsts.SIGMA)
        image = image / 255.0

        return image, landmarks
