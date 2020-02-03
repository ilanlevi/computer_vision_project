import os
import sys
import traceback

import numpy as np
from keras_preprocessing.image import Iterator
from keras_preprocessing.image.utils import array_to_img
from sklearn.model_selection import train_test_split

from consts import BATCH_SIZE, PICTURE_SUFFIX, PICTURE_SIZE, CANNY_SIGMA, CSV_LABELS, CSV_OUTPUT_FILE_NAME
from image_utils import load_image, auto_canny, resize_image_and_landmarks, wrap_roi, clean_noise, landmarks_transform
from image_utils import load_image_landmarks
from my_utils import get_files_list, my_mkdir, write_csv, get_suffix
from .fpn_wrapper import FpnWrapper


# todo -comments
class MyDataIterator(Iterator):

    def __init__(self,
                 data_path,
                 image_data_generator=None,
                 original_file_list=None,
                 fpn_model=FpnWrapper(),
                 batch_size=BATCH_SIZE,
                 picture_suffix=PICTURE_SUFFIX,
                 out_image_size=PICTURE_SIZE,
                 gen_y=False,
                 shuffle=False,
                 sample_weight=None,
                 seed=None,
                 save_to_dir=None,
                 save_csv=False,
                 save_images=False,
                 save_prefix='',
                 should_clean_noise=False,
                 use_canny=False,
                 save_format='png',
                 dtype='float32'):

        self.image_generator = image_data_generator
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
        self.should_clean_noise = should_clean_noise
        self.use_canny = use_canny

        # set grayscale channel
        self.image_shape = (self.out_image_size, self.out_image_size, 1)

        if not isinstance(picture_suffix, list):
            picture_suffix = [picture_suffix]

        self.picture_suffix = picture_suffix

        if original_file_list is None:
            original_file_list = self._get_original_list()

        self.list_of_files = original_file_list
        self.list_of_files = np.asarray(self.list_of_files)
        self.save_csv = save_csv
        self.save_images = save_images

        self.should_save = self.gen_y and (self.save_to_dir is not None)
        self.should_save_aug = self.should_save and self.save_images
        self.should_save_csv = self.should_save and self.save_csv

        super(MyDataIterator, self).__init__(len(self.list_of_files), self.batch_size, self.shuffle, self.seed)

    def set_gen_labels(self, gen_y, save_to_dir=None):
        """Set self.gen_y and self.save_to_dir, update self.should_save_aug as well"""
        self.gen_y = gen_y
        self.save_to_dir = save_to_dir

        self.should_save = self.gen_y and (self.save_to_dir is not None)

        self.should_save_aug = self.should_save and self.save_images
        self.should_save_csv = self.should_save and self.save_csv

    def set_image_data_generator(self, image_data_generator):
        self.image_generator = image_data_generator

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
        self.list_of_files = np.asarray(self.list_of_files)
        self.n = len(self.list_of_files)
        if v_size is not None:
            test_files, validation_files = train_test_split(test_files, test_size=v_size, shuffle=self.shuffle)
        else:
            validation_files = []

        self._set_index_array()
        return test_files, validation_files

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)
        self.index_array = self.list_of_files[self.index_array]

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def get_steps_per_epoch(self):
        return len(self.list_of_files) // self.batch_size

    def _get_batches_of_transformed_samples(self, index_array):
        print('blabla')
        batch_out_shape = tuple([len(index_array)] + list(self.image_shape))

        batch_x = np.zeros(batch_out_shape)
        batch_mask = np.zeros(batch_out_shape)
        batch_y = np.zeros(shape=(len(index_array), 6))

        for i, image_path in enumerate(index_array):
            score = self._get_x_y(image_path)
            batch_x[i], batch_y[i], batch_mask[i] = score

        # save all if needed
        if self.should_save:
            self._save_all_if_needed(batch_x, batch_mask, batch_y, index_array)

        output = (batch_x, batch_y)
        if not self.gen_y:
            return output[0]
        if self.sample_weight is not None:
            output += (self.sample_weight[index_array],)
        return output

    def _get_x_y(self, image_path):
        while True:
            try:
                image, landmarks, y = self._get_samples(image_path)

                # remove noise
                if self.gen_y and self.should_clean_noise:
                    image = clean_noise(image)

                # use canny filter
                if self.use_canny:
                    image = auto_canny(image, CANNY_SIGMA)

                image = image[..., np.newaxis]
                random_params = None
                # only if we want to train the model
                if self.gen_y:
                    if self.image_generator is not None:
                        random_params = self.image_generator.get_random_transform(self.image_shape)
                        image = self.image_generator.apply_transform(image.astype(self.dtype), random_params)
                else:
                    return image, np.zeros(image.shape), np.zeros((1, 6))

                masks = []

                if random_params:
                    landmarks_transformed = landmarks_transform(
                        self.out_image_size,
                        self.out_image_size,
                        landmarks,
                        random_params
                    )

                for j in range(68):
                    new_mask = create_single_landmark_mask(landmarks[j], (self.out_image_size, self.out_image_size))
                    if random_params is not None and self.image_generator is not None:
                        new_mask = new_mask[..., np.newaxis]
                        new_mask = self.image_generator.apply_transform(new_mask.astype(self.dtype), random_params)
                        new_mask = np.squeeze(new_mask, axis=2)
                    masks.append(new_mask)
                masks = np.asarray(masks)

                masks = np.reshape(masks, (68, self.out_image_size, self.out_image_size))
                should_flip = self.image_generator is not None
                new_landmarks = get_landmarks_from_masks(masks, should_flip)

                if new_landmarks is not None and len(new_landmarks) is 68:
                    new_landmarks = np.asarray(new_landmarks, dtype=self.dtype)
                    new_pose = self.fpn_model.get_3d_vectors(new_landmarks)
                    new_pose = np.asarray(new_pose)
                    new_pose = np.resize(new_pose, (1, 6))

                    curr_mask = np.zeros(self.image_shape)
                    # create mask for testing output
                    if self.should_save_aug:
                        curr_mask = create_mask_from_landmarks(new_landmarks, self.image_shape)
                        curr_mask = curr_mask[..., np.newaxis]

                    return image, new_pose, curr_mask

            except Exception as e:
                print('Error happened while reading image: %s! Error: %s' % (image_path, str(e)))
                traceback.print_exc(file=sys.stdout)

    def _save_all_if_needed(self, batch_x, batch_mask, batch_y, index_array):
        """
        Works only if self.should_save_aug is true.
        Save all of the augmented images, masks and out_y in dst dir. (out_y in csv)
        :param batch_x: the batch randomly augmented images
        :param batch_mask: the batch numbered landmarks masks
        :param batch_y: the batch 6DoF values
        :param index_array: this batch indexes array
        """
        # check lengths
        should_write_scores = (len(batch_x) == len(index_array))
        should_write_scores = should_write_scores and self.should_save

        if should_write_scores:
            csv_rows = []

            # create folder if missing
            my_mkdir(self.save_to_dir)

            for i, j in enumerate(index_array):
                rnd = np.random.randint(1e4)
                fname = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=j,
                    hash=rnd,
                    format=self.save_format)
                mask_name = '{prefix}_{index}_{hash}.{format}'.format(
                    prefix=('mask_' + self.save_prefix),
                    index=j,
                    hash=rnd,
                    format=self.save_format)
                if self.save_images:
                    batch_out_shape = tuple([len(batch_x)] + list(self.image_shape))
                    batch_mask = np.reshape(batch_mask, batch_out_shape)

                    img = array_to_img(batch_x[i], 'channels_last', scale=True)
                    img.my_save(os.path.join(self.save_to_dir, fname))
                    mask_img = array_to_img(batch_mask[i], 'channels_last', scale=True)
                    mask_img.my_save(os.path.join(self.save_to_dir, mask_name))

                rx, ry, rz, tx, ty, tz = batch_y[i]
                if self.image_generator is None:
                    fname = get_suffix(j, '\\')
                csv_rows.append([i, fname, rx, ry, rz, tx, ty, tz])

            if self.should_save_csv:
                write_csv(csv_rows, CSV_LABELS, self.save_to_dir, CSV_OUTPUT_FILE_NAME, append=True)

    def _get_samples(self, image_path):
        """
        :param image_path: an image_path from self.list_of_files
        :return: tuple of: (loaded image, landmarks or None if !self.gen_y, 6DoF or None if !self.gen_y)
        """
        image, landmarks = self._load_image(image_path)
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
            image = load_image(image_path)
            landmarks = load_image_landmarks(image_path)
            image = wrap_roi(image, landmarks)
            image, landmarks = resize_image_and_landmarks(image, landmarks, self.out_image_size)
        else:
            image = load_image(image_path, self.out_image_size)
            landmarks = None

        return np.asarray(image), np.asarray(landmarks)
