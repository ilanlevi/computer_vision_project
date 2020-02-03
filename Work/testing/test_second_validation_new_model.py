"""
    This is a testing class for checking validation set scores
"""
import os

import matplotlib.pyplot as plt
import numpy as np

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

    INPUT_SIZE = (PICTURE_SIZE, PICTURE_SIZE)

    # img_gen = ImageDataGenerator()
    pose_datagen = ImagePoseGenerator(folder,
                                      shuffle=False,
                                      batch_size=1,
                                      gen_y=True,
                                      mask_size=INPUT_SIZE,
                                      follow_links=False,
                                      )

    files = pose_datagen.filepaths
    scores = []
    for i, file_name in enumerate(files):
        _, y = pose_datagen.next()
        rx, ry, rz, tx, ty, tz = np.squeeze(y)
        name = get_suffix(file_name, '\\')
        this_score = [i, name, rx, ry, rz, tx, ty, tz]
        scores.append(this_score)

    write_csv(scores, CSV_LABELS, folder, filename_my, False)
    compare_scores(folder, filename_valid, filename_my, filename_diff, False)
    plot_diff(folder, filename_diff, title='diff')

    plot_diff_each_param(folder, [filename_valid, filename_my])

    plt.show()
