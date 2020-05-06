import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import DirectoryIterator, array_to_img, os
from sklearn.model_selection import train_test_split

from consts import OUT_DIM, PICTURE_SIZE, BATCH_SIZE, CSV_LABELS, CSV_OUTPUT_FILE_NAME
from image_utils import load_image, wrap_roi, \
    pose_transform, flip_landmarks, my_resize, load_image_landmarks, draw_landmarks_axis
from image_utils.fpn_wrapper import FpnWrapper, rotation_2_euler, euler_2_rotation
from my_utils import my_mkdir, write_csv, get_suffix


class ImagePoseGenerator(DirectoryIterator):

    def __init__(self,
                 directory,
                 image_data_generator=ImageDataGenerator(),
                 f_list=None,
                 fpn_model=FpnWrapper(),
                 image_shape=(PICTURE_SIZE, PICTURE_SIZE),
                 batch_size=BATCH_SIZE,
                 shuffle=True,
                 csv_name=CSV_OUTPUT_FILE_NAME,
                 to_gray=False,
                 gen_y=False,
                 seed=None,
                 save_to_dir=None,
                 color_mode='grayscale',
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 save_images=False,
                 follow_links=False,
                 interpolation='nearest',
                 d_type='float32'):
        """
        Create DirectoryIterator
        :param directory: the directory to go on
        :param image_data_generator: the image generator, if none specified will use default value (no augmentation)
        :param f_list: list of files (instead of the directory param)
        :param fpn_model: the fpn_model, if none specified will use default value (see consts)
        :param image_shape: the image shape
        :param batch_size: batch size
        :param shuffle: to shuffle files or not
        :param csv_name: the csv output name (if required to save)
        :param to_gray: convert images to grayscale or not
        :param gen_y: generate labels or not
        :param seed: the seed
        :param save_to_dir: save output to dir name
        :param color_mode: the color mode
        :param save_prefix: save images prefix format
        :param save_format: save images format
        :param save_images: save augmentation images or not
        """

        self.fpn_model = fpn_model
        self.mask_size = image_shape
        self.gen_y = gen_y
        self.to_gray = to_gray
        self.save_images = save_images
        self.csv_name = csv_name

        super(ImagePoseGenerator, self).__init__(directory,
                                                 image_data_generator,
                                                 image_shape,
                                                 color_mode=color_mode,
                                                 classes=None,
                                                 class_mode='categorical',
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 seed=seed,
                                                 data_format='channels_last',
                                                 save_to_dir=save_to_dir,
                                                 save_prefix=save_prefix,
                                                 save_format=save_format,
                                                 follow_links=follow_links,
                                                 subset=subset,
                                                 interpolation=interpolation,
                                                 dtype=d_type)
        if f_list is not None:
            self._filepaths = f_list
            self.filenames = f_list
            self.samples = len(f_list)
            self.n = self.samples
            print('Now I Found %d images belonging to %d classes.' % (self.samples, self.num_classes))

    def split_to_train_and_validation(self, t_size, v_size=None):
        """
        Split arrays or matrices into random train, test and validation subsets.
        Updates self.original_file_list as well (to test value)
        :param t_size: should be [0.0 ,1.0]. represent the proportion of the dataset to include in the test split split
        :param v_size: should be [0.0 ,1.0]. represent the proportion of the dataset to include in the validation split
        :return: tuple (test_files, validation_files).
        """
        self._filepaths, test_files = train_test_split(self._filepaths, test_size=t_size, shuffle=self.shuffle)
        self.filenames = self._filepaths
        self.samples = len(self.filenames)
        self.n = self.samples
        print('Now I Found %d images belonging to %d classes.' % (self.samples, self.num_classes))
        if v_size is not None:
            test_files, validation_files = train_test_split(test_files, test_size=v_size, shuffle=self.shuffle)
        else:
            validation_files = []
        return test_files, validation_files

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: Array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        # save other params
        image_generator = self.image_data_generator
        if image_generator is not None:
            random_params = image_generator.get_random_transform(self.image_shape)
        else:
            random_params = None

        # build batch of image data
        batch_x = np.zeros(tuple([len(index_array)] + list(self.image_shape)), dtype=self.dtype)
        batch_y = np.zeros((len(index_array), OUT_DIM), dtype=self.dtype)

        for i, j in enumerate(index_array):
            original, original_pose = self._load_data(filepaths[j], random_params)

            if self.to_gray:
                # add grayscale axis if needed
                original = original[..., np.newaxis]

            if random_params is not None:  # if random_params is not None -> image_generator is not None
                x = image_generator.apply_transform(original, random_params)
                x = image_generator.standardize(x)

                if self.gen_y:
                    r_vec = original_pose[:3]

                    euler_vector = rotation_2_euler(r_vec)
                    euler_vector = pose_transform(euler_vector, random_params)
                    score = euler_2_rotation(euler_vector)

                    score = np.asarray(score)

                    batch_y[i] = score

            else:
                # something failed, use original
                x = image_generator.standardize(original)
                pose_vector = np.append(original_pose[:3])
                batch_y[i] = pose_vector

            x = np.squeeze(x)
            x = np.reshape(x, self.image_shape)
            batch_x[i] = x

        if not self.gen_y:  # check if we need to create batch_y or not
            return batch_x

        self._save_all_if_needed(batch_x.copy(), batch_y.copy(), filepaths.copy())

        return batch_x, batch_y

    def _load_data(self, image_path, random_params):
        """
        Load images and pose if self.gen_y. Also does flip if needed.
        Will return an image in the proper size and only the expanded roi box of the face
        :param image_path: the image path to load
        :param random_params: the random param (for flipping horizontal)
        :return: tuple of (image, pose) and if not self.gen_y: (image, None)
        """
        image = load_image(image_path, gray=self.to_gray)

        if not self.gen_y:
            image = my_resize(image, self.mask_size[0], self.mask_size[1])
            return image, None

        landmarks = load_image_landmarks(image_path)

        if random_params is not None and random_params.get('flip_horizontal', 0):
            landmarks = flip_landmarks(landmarks, image.shape[1])
            image = cv2.flip(image, 1)

        pose = self.fpn_model.get_3d_vectors(landmarks)
        image = wrap_roi(image, landmarks)
        image = my_resize(image, self.mask_size[0], self.mask_size[0])

        return image, pose

    def _save_all_if_needed(self, batch_x, batch_y, file_names):
        """
        Works only if self.should_save_aug is true.
        Save all of the augmented images and out_y in dst dir. (out_y in csv)
        :param batch_x: the batch randomly augmented images
        :param batch_y: the batch 6DoF values
        :param file_names: this batch files path array
        """
        if self.save_to_dir is not None:
            csv_rows = []
            new_path = os.path.join(self.directory, self.save_to_dir)
            # create folder if missing
            if self.save_images:
                my_mkdir(new_path)
            new_path = new_path + '\\'

            for i in range(len(batch_x)):
                file_name = file_names[i]
                file_name = get_suffix(file_name, '\\')
                rnd = np.random.randint(1e4)

                rx, ry, rz = batch_y[i]

                f_name = '{index}_{prefix}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=(new_path + file_name),
                    hash=rnd,
                    format=self.save_format)

                if self.save_images:
                    rotation_vector = np.asarray([rx, ry, rz])
                    img = draw_landmarks_axis(batch_x[i], rotation_vector)
                    img = array_to_img(img, self.data_format, scale=False)
                    img.save(f_name)

                f_name = get_suffix(f_name, '\\')
                csv_rows.append([i, f_name, rx, ry, rz, 0, 0, 0])

                write_csv(csv_rows, CSV_LABELS, self.directory, self.csv_name, append=True)
