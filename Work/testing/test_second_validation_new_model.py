"""
    This is a testing class for checking validation set scores
"""
import os

import matplotlib.pyplot as plt

from compare_utils import compare_scores, plot_diff, plot_diff_each_param
from consts import VALIDATION_DIFF_CSV, PICTURE_NAME, \
    VALIDATION_CSV_2, VALIDATION_FOLDER_2, CSV_OUTPUT_FILE_NAME
from my_models import FpnWrapper, MyDataIterator
from my_utils import read_csv

if __name__ == '__main__':
    fpn = FpnWrapper()

    folder = VALIDATION_FOLDER_2

    filename_my = CSV_OUTPUT_FILE_NAME
    filename_valid = VALIDATION_CSV_2
    filename_diff = VALIDATION_DIFF_CSV

    os.remove(folder + filename_my)
    csv = read_csv(folder, filename_valid)
    file_list = [(folder + r.get(PICTURE_NAME)) for r in csv]

    my_iterator = MyDataIterator(folder, None, save_to_dir=folder, shuffle=False, gen_y=True, save_csv=True,
                                 original_file_list=file_list)
    for i in range(len(my_iterator)):
        my_iterator.next()

    compare_scores(folder, filename_valid, filename_my, filename_diff, False)
    plot_diff(folder, filename_diff, title='diff')

    plot_diff_each_param(folder, [filename_valid, filename_my])

    plt.show()
