"""
    This is a testing class for checking validation set scores
"""

import matplotlib.pyplot as plt
import numpy as np

from compare_utils import compare_scores, plot_diff, plot_diff_each_param
from consts import MY_VALIDATION_CSV, VALIDATION_DIFF_CSV, CSV_LABELS, PICTURE_NAME, \
    VALIDATION_CSV_2, VALIDATION_FOLDER_2
from my_models import FpnWrapper
from my_utils import write_csv, read_csv, get_prefix


def align_images(fpm_model, csv_lines):
    scores_vectors = []

    for i, row in enumerate(csv_lines):
        landmarks = np.zeros((68, 2))
        name = row.get(PICTURE_NAME)

        string_to_save = 'version: 1\nn_points:  68\n{\n'
        for j in range(68):
            xy = row.get(str(j))
            xy = xy[1:-1]
            string_to_save = string_to_save + xy + '\n'
            landmarks[j] = np.fromstring(xy, dtype=float, sep=' ')
        string_to_save = string_to_save + '}\n'

        landmarks = np.asarray(landmarks)
        landmarks = np.reshape(landmarks, (68, 2))

        rx, ry, rz, tx, ty, tz = fpm_model.get_3d_vectors(landmarks)
        this_score = [i, name, rx, ry, rz, tx, ty, tz]
        scores_vectors.append(this_score)

        pts_name = VALIDATION_FOLDER_2 + get_prefix(name) + '.pts'

        with open(pts_name, 'w') as out_f:
            out_f.writelines(string_to_save)

    return scores_vectors


def write_scores(folder_path, filename, fpn_model, csv_lines, print_scores=False):
    s = align_images(fpn_model, csv_lines)
    write_csv(s, CSV_LABELS, folder_path, filename, print_scores)


if __name__ == '__main__':
    fpn = FpnWrapper()

    folder = VALIDATION_FOLDER_2

    filename_my = MY_VALIDATION_CSV
    filename_valid = VALIDATION_CSV_2
    filename_diff = '2' + VALIDATION_DIFF_CSV

    csv = read_csv(folder, filename_valid)

    write_scores(folder, MY_VALIDATION_CSV, fpn, csv)

    compare_scores(folder, filename_valid, filename_my, filename_diff, False)
    plot_diff(folder, filename_diff, title='diff')

    plot_diff_each_param(folder, [filename_valid, filename_my])

    plt.show()
