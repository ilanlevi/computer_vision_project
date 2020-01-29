from keras_preprocessing.image import ImageDataGenerator, os

from consts import DataSetConsts as dsConsts
from generators import MyDataIterator
from mytools import get_files_list

"""Please ignore! This will be used for testing LandmarkWrapper, FpnWrapper classes"""

if __name__ == '__main__':
    folder = 'C:\\Work\\ComputerVision\\Project\\tmp\\'

    # clean up shit
    to_remove = get_files_list(folder + 'out\\')
    print(to_remove)
    for remove_file in to_remove:
        os.remove(remove_file)

    suffixes = dsConsts.PICTURE_SUFFIX
    images = get_files_list(folder, suffixes, [dsConsts.LANDMARKS_FILE_SUFFIX, dsConsts.LANDMARKS_PREFIX])

    datagen = ImageDataGenerator(
        shear_range=10,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True, data_format='channels_first')

    iterator = MyDataIterator(folder, datagen, save_to_dir=(folder + '\\out\\'), gen_y=True, out_image_size=250,
                              batch_size=5)
    datagen.fit(iterator, rounds=1)
