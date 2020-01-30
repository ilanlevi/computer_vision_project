from keras_preprocessing.image import ImageDataGenerator, os

from consts import DataSetConsts as dsConsts
from generators import MyDataIterator
from mytools import get_files_list

"""Please ignore! This will be used for testing LandmarkWrapper, FpnWrapper classes"""

if __name__ == '__main__':
    folder = 'C:\\Work\\ComputerVision\\Project\\tmp\\'

    # clean up shit
    to_remove = get_files_list(folder + 'out_new\\')
    for remove_file in to_remove:
        os.remove(remove_file)

    suffixes = dsConsts.PICTURE_SUFFIX
    images_list = get_files_list(folder, suffixes, ['out'])
    data_format = 'channels_first'
    datagen = ImageDataGenerator(
        shear_range=10,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True, data_format=data_format)

    iterator = MyDataIterator(folder, datagen, original_file_list=images_list, save_to_dir=(folder + '\\out_new\\'),
                              gen_y=True, out_image_size=250,
                              batch_size=4)
    datagen.flow(iterator)
