import numpy as np
from keras_preprocessing.image import DirectoryIterator
from keras_preprocessing.image.utils import array_to_img, img_to_array

from image_utils import load_image, resize_image_and_landmarks
from image_utils import load_image_landmarks, get_landmarks_from_masks, create_single_landmark_mask
from .fpn_wrapper import FpnWrapper


# todo -comments
class ImagePoseGenerator(DirectoryIterator):

    def __init__(self,
                 directory,
                 image_data_generator,
                 fpn_model=FpnWrapper(),
                 mask_size=(256, 256),
                 batch_size=32,
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
        gen_y = self.gen_y
        image_generator = self.image_data_generator
        if image_generator is not None:
            params = self.image_data_generator.get_random_transform(self.image_shape)
        else:
            params = None
        # build batch of image data
        batch_x = np.zeros(tuple([len(index_array)] + list(self.image_shape)), dtype=self.dtype)
        batch_y = np.zeros((len(index_array), 6), dtype=self.dtype)

        for i, j in enumerate(index_array):
            original, image_landmarks = self._load_landmark(filepaths[j])
            original = original[..., np.newaxis]

            if params is not None:  # if params is not None -> image_generator is not None
                x = image_generator.apply_transform(original, params)
                x = image_generator.standardize(x)
                new_landmarks = self.transform_landmarks_mask(image_landmarks, self.image_shape, image_generator,
                                                              params)
            else:
                # we cannot create new image, use the original image
                new_landmarks = None
                x = original

            if new_landmarks is None:
                #
                new_landmarks = image_landmarks
                if image_generator is not None:
                    x = self.image_data_generator.standardize(original)
                else:
                    x = original

            if self.gen_y:
                pose = self.fpn_model.get_3d_vectors(new_landmarks)
                batch_y[i] = pose

            x = np.squeeze(x)
            x = np.reshape(x, self.image_shape)
            batch_x[i] = x

        if not gen_y:  # check if we need to create batch_y or not
            return batch_x

        return batch_x, batch_y

    def transform_landmarks_mask(self, landmarks, mask_size, image_generator, params):
        masks_before = np.stack([np.array(create_single_landmark_mask(landmark, mask_size)) for landmark in landmarks])
        masks_after = np.stack([np.array(image_generator.apply_transform
                                         (img_to_array(array_to_img(mask, data_format=self.data_format),
                                                       data_format=self.data_format), params)) for
                                mask in masks_before[:]])
        masks_after = np.reshape(masks_after, tuple([68] + list(mask_size)))
        new_landmarks = get_landmarks_from_masks(masks_after, flip_back=True)
        return new_landmarks

    def _load_landmark(self, image_path):
        """
        :param image_path: path to image to load
        :return: tuple of: (loaded image after canny, landmarks or None if !self.gen_y)
        """

        image = load_image(image_path)
        landmarks = load_image_landmarks(image_path)
        image, landmarks = resize_image_and_landmarks(image, landmarks, self.mask_size[0])

        return image, landmarks
