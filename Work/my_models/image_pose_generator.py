import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import DirectoryIterator, array_to_img, os

from consts import OUT_DIM, PICTURE_SIZE, BATCH_SIZE, CSV_LABELS, CSV_OUTPUT_FILE_NAME
from image_utils import load_image, resize_image_and_landmarks, landmarks_transform, create_numbered_mask
from image_utils import load_image_landmarks
from my_utils import my_mkdir, write_csv, get_suffix
from .fpn_wrapper import FpnWrapper


# todo -comments
class ImagePoseGenerator(DirectoryIterator):

    def __init__(self,
                 directory,
                 image_data_generator=ImageDataGenerator(),
                 fpn_model=FpnWrapper(),
                 mask_size=(PICTURE_SIZE, PICTURE_SIZE),
                 batch_size=BATCH_SIZE,
                 shuffle=True,
                 gen_y=False,
                 seed=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 follow_links=False,
                 interpolation='nearest',
                 dtype='float32'):
        self.fpn_model = fpn_model
        self.mask_size = mask_size
        self.gen_y = gen_y

        super(ImagePoseGenerator, self).__init__(directory,
                                                 image_data_generator,
                                                 mask_size,
                                                 color_mode='grayscale',
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
                                                 dtype=dtype)

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
        gen_y = self.gen_y
        image_generator = self.image_data_generator
        if image_generator is not None:
            random_params = self.image_data_generator.get_random_transform(self.image_shape)
        else:
            random_params = None

        # build batch of image data
        batch_x = np.zeros(tuple([len(index_array)] + list(self.image_shape)), dtype=self.dtype)
        masks = np.zeros(tuple([len(index_array)] + list(self.image_shape)), dtype=self.dtype)
        batch_y = np.zeros((len(index_array), OUT_DIM), dtype=self.dtype)

        for i, j in enumerate(index_array):
            original, image_landmarks = self._load_landmark(filepaths[j])
            original = original[..., np.newaxis]

            if random_params is not None:  # if random_params is not None -> image_generator is not None
                x = image_generator.apply_transform(original, random_params)
                x = image_generator.standardize(x)
                new_landmarks = self.transform_landmarks(image_landmarks, self.mask_size,
                                                         random_params)
            else:
                # we cannot create the new image landmarks, use the original image
                new_landmarks = None
                x = original

            if new_landmarks is None:
                print(filepaths[j])
                new_landmarks = image_landmarks
                x = self.image_data_generator.standardize(original)
            else:
                print('Bla')

            if self.gen_y:
                pose = self.fpn_model.get_3d_vectors(new_landmarks)
                batch_y[i] = pose
                if self.save_to_dir is not None:
                    masks[i] = create_numbered_mask(new_landmarks, self.image_shape)

            x = np.squeeze(x)
            x = np.reshape(x, self.image_shape)
            batch_x[i] = x

        if not self.gen_y:  # check if we need to create batch_y or not
            return batch_x

        self._save_all_if_needed(batch_x.copy(), masks.copy(), batch_y.copy(), filepaths.copy())

        return batch_x, batch_y

    @staticmethod
    def transform_landmarks(landmarks, mask_size, params):
        landmarks_transformed = landmarks_transform(
            mask_size[0],
            mask_size[1],
            landmarks,
            params
        )

        # check if all of the landmarks are inside the image
        for landmark in landmarks_transformed:
            if (landmark[0] < 0) or (landmark[0] >= mask_size[1]) or (landmark[1] < 0) or (landmark[1] >= mask_size[0]):
                # when found 1 point out of bound, return None
                return None

        return landmarks_transformed[:, 0:2]

    def _load_landmark(self, image_path):
        """
        :param image_path: path to image to load
        :return: tuple of: (loaded image after canny, landmarks or None if !self.gen_y)
        """

        image = load_image(image_path)
        landmarks = load_image_landmarks(image_path)
        image, landmarks = resize_image_and_landmarks(image, landmarks, self.mask_size[0])

        return image, landmarks

    def _save_all_if_needed(self, batch_x, batch_mask, batch_y, file_names):
        """
        Works only if self.should_save_aug is true.
        Save all of the augmented images, masks and out_y in dst dir. (out_y in csv)
        :param batch_x: the batch randomly augmented images
        :param batch_mask: the batch numbered landmarks masks
        :param batch_y: the batch 6DoF values
        :param file_names: this batch files path array
        """
        # check lengths

        if self.save_to_dir is not None:
            csv_rows = []
            new_path = os.path.join(self.directory, self.save_to_dir)
            # create folder if missing
            my_mkdir(new_path)
            new_path = new_path + '\\'

            for i in range(len(batch_x)):
                file_name = file_names[i]
                file_name = get_suffix(file_name, '\\')
                rnd = np.random.randint(1e4)
                fname = '{index}_{prefix}_{hash}.{format}'.format(
                    prefix=self.save_prefix,
                    index=(new_path + file_name),
                    hash=rnd,
                    format=self.save_format)
                mask_name = '{index}_{prefix}_{hash}.{format}'.format(
                    prefix=('mask_' + self.save_prefix),
                    index=(new_path + file_name),
                    hash=rnd,
                    format=self.save_format)

                img = array_to_img(batch_x[i], self.data_format, scale=False)
                img.save(fname)
                mask_img = array_to_img(batch_mask[i], self.data_format, scale=False)
                mask_img.save(mask_name)

                rx, ry, rz, tx, ty, tz = batch_y[i]
                fname = get_suffix(fname, '\\')
                csv_rows.append([i, fname, rx, ry, rz, tx, ty, tz])

            write_csv(csv_rows, CSV_LABELS, self.directory, CSV_OUTPUT_FILE_NAME, append=False)
