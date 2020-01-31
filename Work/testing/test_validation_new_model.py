"""
    This is a testing class for checking validation set scores
"""
import matplotlib.pyplot as plt

from compare_utils import compare_scores, plot_diff, plot_diff_each_param
from consts import VALIDATION_FOLDER, VALIDATION_CSV, VALIDATION_DIFF_CSV, CSV_OUTPUT_FILE_NAME, PICTURE_NAME
from my_models import FpnWrapper, MyDataIterator
from my_utils import read_csv

if __name__ == '__main__':
    fpn = FpnWrapper()

    folder = VALIDATION_FOLDER
    images_folder = folder + '\\images\\'
    filename_my = CSV_OUTPUT_FILE_NAME
    filename_valid = VALIDATION_CSV
    filename_diff = '1' + VALIDATION_DIFF_CSV

    csv = read_csv(folder, filename_valid)
    file_list = [(images_folder + r.get(PICTURE_NAME)) for r in csv]

    my_iterator = MyDataIterator(folder, None, save_to_dir=folder, shuffle=False, gen_y=True, save_csv=True,
                                 original_file_list=file_list)
    for i in range(len(my_iterator)):
        my_iterator.next()

    compare_scores(folder, filename_valid, filename_my, filename_diff, False)
    plot_diff(folder, filename_diff, title='diff')

    plot_diff_each_param(folder, [filename_valid, filename_my])

    plt.show()
