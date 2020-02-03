"""
    This is a testing class for checking random face augmentation images and landmarks generation
"""
import os

from keras.preprocessing.image import ImageDataGenerator

from consts import VALIDATION_DIFF_CSV, VALIDATION_CSV_2, CSV_OUTPUT_FILE_NAME, PICTURE_SIZE
from my_models import ImagePoseGenerator
from my_utils import get_files_list

NUMBER_OF_BATCHES = 6

if __name__ == '__main__':

    # folder = VALIDATION_FOLDER_2
    folder = 'C:/Work/ComputerVision/valid_set/New folder/'
    save_to_dir = 'testAug'

    filename_my = CSV_OUTPUT_FILE_NAME
    filename_valid = VALIDATION_CSV_2
    filename_diff = VALIDATION_DIFF_CSV

    try:
        # clean old
        files = get_files_list(folder + save_to_dir + '/')
        for f in files:
            os.remove(f)
        os.removedirs(folder + save_to_dir + '/')
    except Exception as e:
        print('Couldn\'t remove dir or files! Error: ' + str(e))

    INPUT_SIZE = (PICTURE_SIZE, PICTURE_SIZE)

    # img_gen = ImageDataGenerator()
    img_gen = ImageDataGenerator(shear_range=5,
                                 rotation_range=30,
                                 # width_shift_range=10,
                                 # height_shift_range=10,
                                 horizontal_flip=True,
                                 )
    pose_datagen = ImagePoseGenerator(folder,
                                      img_gen,
                                      shuffle=True,
                                      batch_size=100,
                                      gen_y=True,
                                      mask_size=INPUT_SIZE,
                                      follow_links=False,
                                      save_to_dir='testAug'
                                      )

    # create 5 batch of random images
    for i in range(NUMBER_OF_BATCHES):
        pose_datagen.next()

    print('Done!')
