import os

from keras_preprocessing.image import ImageDataGenerator

from consts import PICTURE_SUFFIX
from my_models import MyDataIterator
from my_utils import get_files_list

"""Please ignore! This will be used for testing LandmarkWrapper, FpnWrapper classes"""

ROUNDS = 5

if __name__ == '__main__':
    folder = 'C:\\Work\\ComputerVision\\Project\\tmp\\'

    # clean up shit
    to_remove = get_files_list(folder + 'out_new\\')
    for remove_file in to_remove:
        os.remove(remove_file)

    images_list = get_files_list(folder, PICTURE_SUFFIX, ['out'])
    data_format = 'channels_last'
    datagen = ImageDataGenerator(
        shear_range=20,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        data_format=data_format)

    my_iterator = MyDataIterator(folder, datagen, save_to_dir=(folder + '\\out_new\\'),
                                 gen_y=True, save_csv=True, save_images=True, out_image_size=250)

    for i in range(ROUNDS):
        my_iterator.next()

    print('Exiting...')
    exit(1)
