import os

import numpy as np
from keras_preprocessing.image import Iterator
from keras_preprocessing.image.utils import array_to_img
from sklearn.model_selection import train_test_split

from consts import DataSetConsts, CsvConsts, ValidationFileConsts as fConsts
from image_utils import load_image, auto_canny, resize_image_and_landmarks
from mytools import get_files_list, load_image_landmarks, mkdir, \
    get_landmarks_from_masks, create_landmark_mask, write_csv, create_mask_from_landmarks
from .fpn_wrapper_model import MyFpnWrapper


class MyDataIterator(Iterator):

    def __init__(self,
                 data_path,
                 image_data_generator,
                 original_file_list=None,
                 fpn_model=MyFpnWrapper(),
                 batch_size=DataSetConsts.BATCH_SIZE,
                 picture_suffix=DataSetConsts.PICTURE_SUFFIX,
                 out_image_size=DataSetConsts.PICTURE_SIZE,
                 gen_y=False,
                 shuffle=False,
                 sample_weight=None,
                 seed=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 dtype='float32'):

        self.image_data_generator = image_data_generator
        self.data_path = data_path
        self.batch_size = batch_size
        self.gen_y = gen_y
        self.shuffle = shuffle
        self.sample_weight = sample_weight
        self.seed = seed
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.dtype = dtype
        self.out_image_size = out_image_size
        self.fpn_model = fpn_model
        self.im_size = (self.out_image_size, self.out_image_size)

        # set grayscale channel
        self.image_shape = (1, self.out_image_size, self.out_image_size)

        if not isinstance(picture_suffix, list):
            picture_suffix = [picture_suffix]

        self.picture_suffix = picture_suffix

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
        batch_x = np.asarray([])
        batch_mask = np.asarray([])
        batch_y = np.asarray([])

        while len(batch_x) < len(index_array):
            next_i = len(batch_x)
            image_path = index_array[next_i]
            image, landmarks, y = self._get_samples(image_path)
            image = np.reshape(image, self.image_shape)
            # todo - check if noise is added
            random_params = self.image_data_generator.get_random_transform(self.image_shape)
            image = self.image_data_generator.apply_transform(image.astype(self.dtype), random_params)
            image = self.image_data_generator.standardize(image)
            image = np.reshape(image, (self.out_image_size, self.out_image_size))
            image = self._do_canny(image)
            image = np.reshape(image, self.image_shape)
            if self.gen_y:
                masks = []
                for index in range(len(landmarks)):
                    mask = create_landmark_mask(landmarks[index], self.im_size)
                    mask = np.reshape(mask, self.image_shape)
                    mask = self.image_data_generator.apply_transform(mask.astype(self.dtype), random_params)
                    mask = np.reshape(mask, self.im_size)
                    masks.append(mask)

                new_landmarks = get_landmarks_from_masks(masks)

                if new_landmarks is not None and len(new_landmarks) is 68:
                    new_landmarks = np.asarray(new_landmarks, dtype=self.dtype)
                    new_pose = self.fpn_model.get_3d_vectors(new_landmarks)
                    new_pose = np.asarray(new_pose)
                    new_pose = np.resize(new_pose, (1, 6))
                    if len(batch_y) is 0:
                        batch_y = new_pose
                    else:
                        batch_y = np.append(batch_y, new_pose, axis=0)
                    if len(batch_mask) is 0:
                        batch_mask = create_mask_from_landmarks(new_landmarks, self.image_shape)
                    else:
                        batch_mask = np.append(batch_mask, create_mask_from_landmarks(new_landmarks, self.image_shape),
                                               axis=0)
                    # todo delete
                    # tmp_delete[i] = create_landmark_image(new_landmarks, self.image_shape[1:], dtype=self.dtype)
                    # tmp_delete[i] = create_numbered_mask(new_landmarks, self.im_size)
            if len(batch_x) is 0:
                batch_x = image
            else:
                batch_x = np.append(batch_x, image, axis=0)

        # now add channel

        batch_x = np.reshape(batch_x, (len(index_array), self.image_shape[0], self.image_shape[1], self.image_shape[2]))
        if self.save_to_dir is not None:
            batch_mask = np.reshape(batch_mask,
                                    (len(index_array), self.image_shape[0], self.image_shape[1], self.image_shape[2]))
            # create folder if missing
            mkdir(self.save_to_dir)
            for i, j in enumerate(index_array):
                img = array_to_img(batch_x[i], 'channels_first', scale=True)
                rnd = np.random.randint(1e4)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=rnd,
                    format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
                if self.gen_y:
                    mask_name = '{prefix}_{index}_{hash}.{format}'.format(
                        prefix=('mask_' + self.save_prefix),
                        index=j,
                        hash=rnd,
                        format=self.save_format)
                    mask_img = array_to_img(batch_mask[i] + 1, 'channels_first', scale=True)
                    mask_img.save(os.path.join(self.save_to_dir, mask_name))
            csv_rows = []
            for i, (rx, ry, rz, tx, ty, tz) in enumerate(batch_y):
                csv_rows.append([i, self.list_of_files[i], rx, ry, rz, tx, ty, tz])
            write_csv(csv_rows, CsvConsts.CSV_LABELS, self.save_to_dir, fConsts.MY_VALIDATION_CSV)

        output = (batch_x, batch_y)
        if not self.gen_y:
            return output[0]
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output

    def _get_samples(self, index):
        """
        :param index: index from self.list_of_files
        :return: tuple of: (loaded image, landmarks or None if !self.gen_y, 6DoF or None if !self.gen_y)
        """
        image, landmarks = self._load_image(self.list_of_files[index])
        y = None
        if self.gen_y:
            y = self.fpn_model.get_3d_vectors(landmarks)
            y = np.asarray(y, dtype=np.float)
        return image, landmarks, y

    def _load_image(self, image_path):
        """
        :param image_path: path to image to load
        :return: tuple of: (loaded image after canny, landmarks or None if !self.gen_y)
        """
        if self.gen_y:
            # todo - crop face on image and mask
            image = load_image(image_path)
            landmarks = load_image_landmarks(image_path)
            image, landmarks = resize_image_and_landmarks(image, landmarks, self.out_image_size)
        else:
            image = load_image(image_path, self.out_image_size)
            landmarks = None

        return image, landmarks

    @staticmethod
    def _do_canny(image):
        image = np.asarray(image, dtype=np.uint8)
        image = auto_canny(image, DataSetConsts.SIGMA)
        return image
