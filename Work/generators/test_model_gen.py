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
    data_format = 'channels_first'
    datagen = ImageDataGenerator(
        shear_range=10,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True, data_format=data_format)
    # datagen = ImageDataGenerator(rotation_range=20,
    #                              width_shift_range=10.0,
    #                              height_shift_range=10.0,
    #                              ## Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
    #                              shear_range=5.0,
    #                              ## zoom_range: Float or [lower, upper].
    #                              ## Range for random zoom. If a float,
    #                              ## [lower, upper] = [1-zoom_range, 1+zoom_range]
    #                              zoom_range=[0.6, 1.2],
    #                              fill_mode='nearest',
    #                              # cval=-2,
    #                              horizontal_flip=True,
    #                              vertical_flip=False)

    iterator = MyDataIterator(folder, datagen, save_to_dir=(folder + '\\out\\'), gen_y=True, out_image_size=250,
                              batch_size=4, data_format=data_format)
    datagen.flow(iterator)
