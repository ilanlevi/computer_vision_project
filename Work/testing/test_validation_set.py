"""
    This is a testing class for checking validation set scores
"""
import matplotlib.pyplot as plt

from compare_utils import compare_scores, plot_diff, plot_diff_each_param
from consts import VALIDATION_FOLDER, MY_VALIDATION_CSV, VALIDATION_CSV, VALIDATION_DIFF_CSV, CSV_LABELS, PICTURE_NAME
from image_utils import FpnWrapper
from my_utils import write_csv, get_suffix, read_csv
from testing import LabeledDataStore


def align_images(fpm_model, data_set):
    scores_vectors = []

    for i in range(len(data_set.original_file_list)):
        name = get_suffix(data_set.original_file_list[i], '\\')

        rx, ry, rz, tx, ty, tz = fpm_model.get_3d_vectors(data_set.y[i])
        this_score = [i, name, rx, ry, rz, tx, ty, tz]

        scores_vectors.append(this_score)

    return scores_vectors


def write_scores(folder_path, filename, fpn_model, data_set, print_scores=False):
    s = align_images(fpn_model, data_set)
    write_csv(s, CSV_LABELS, folder_path, filename, print_scores)


if __name__ == '__main__':
    fpn = FpnWrapper()

    folder = VALIDATION_FOLDER
    images_folder = folder + '\\images\\'
    filename_my = MY_VALIDATION_CSV
    filename_valid = VALIDATION_CSV
    filename_diff = VALIDATION_DIFF_CSV

    csv = read_csv(folder, filename_valid)
    file_list = [(images_folder + r.get(PICTURE_NAME)) for r in csv]

    data = LabeledDataStore(data_path=VALIDATION_FOLDER, original_file_list=file_list).read_data_set()

    write_scores(VALIDATION_FOLDER, MY_VALIDATION_CSV, fpn, data)

    compare_scores(folder, filename_valid, filename_my, filename_diff, False)
    plot_diff(folder, filename_diff, title='diff')

    plot_diff_each_param(folder, [filename_valid, filename_my])

    plt.show()
