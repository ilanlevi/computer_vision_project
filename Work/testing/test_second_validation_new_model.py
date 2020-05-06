"""
    This is a testing class for checking validation set scores
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

from compare_utils import compare_scores, plot_diff, plot_diff_each_param
from consts import VALIDATION_DIFF_CSV, VALIDATION_CSV_2, CSV_OUTPUT_FILE_NAME, PICTURE_SIZE, CSV_LABELS
from my_models import ImagePoseGenerator
from my_utils import get_suffix, write_csv

if __name__ == '__main__':

    # folder = VALIDATION_FOLDER_2
    folder = 'C:/Work/ComputerVision/valid_set/New folder/'

    filename_my = CSV_OUTPUT_FILE_NAME
    filename_valid = VALIDATION_CSV_2
    filename_diff = VALIDATION_DIFF_CSV

    if os.path.exists(folder + filename_my):
        os.remove(folder + filename_my)

    if os.path.exists(folder + 'testAug'):
        files = glob.glob(folder + 'testAug/*')
        for f in files:
            os.remove(f)

    INPUT_SIZE = (PICTURE_SIZE, PICTURE_SIZE)

    img_gen = ImageDataGenerator()
    pose_datagen = ImagePoseGenerator(folder,
                                      img_gen,
                                      shuffle=False,
                                      batch_size=1,
                                      to_gray=False,
                                      color_mode='rgb',
                                      gen_y=True,
                                      image_shape=INPUT_SIZE,
                                      follow_links=False,
                                      save_to_dir='testAug',
                                      save_images=True
                                      )

    files = pose_datagen.filepaths

    scores = []
    for i, file_name in enumerate(files):
        _, y = pose_datagen.next()
        rx, ry, rz = np.squeeze(y)
        name = get_suffix(file_name, '\\')
        this_score = [i, name, rx, ry, rz, 0, 0, 0]
        scores.append(this_score)

    write_csv(scores, CSV_LABELS, folder, filename_my, False)
    compare_scores(folder, filename_valid, filename_my, filename_diff, False)
    plot_diff(folder, filename_diff, title='diff')

    plot_diff_each_param(folder, [filename_valid, filename_my])

    plt.show()
