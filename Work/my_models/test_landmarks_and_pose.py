import os

import numpy as np

from consts import DataSetConsts as dsConsts, ValidationFileConsts as fConsts, CsvConsts
from image_utils import load_image
from my_models import LandmarkWrapper, FpnWrapper
from mytools import get_files_list, write_csv, read_csv

"""Please ignore! This will be used for testing LandmarkWrapper, FpnWrapper classes"""

if __name__ == '__main__':
    folder = 'C:\\Work\\ComputerVision\\Project\\tmp\\'

    suffixes = dsConsts.PICTURE_SUFFIX
    images = get_files_list(folder, suffixes, [dsConsts.LANDMARKS_FILE_SUFFIX, dsConsts.LANDMARKS_PREFIX])

    lm_wrap = LandmarkWrapper(save_to_dir=True)
    fpn_wrap = FpnWrapper()

    score = []
    for im in images:
        lmarks = lm_wrap.load_image_landmarks(im)
        rx, ry, rz, tx, ty, tz = fpn_wrap.get_3d_vectors(lmarks)
        score.append([rx, ry, rz, tx, ty, tz])

        tmp_img = load_image(im)
        shape = np.shape(tmp_img)
        shape = max(shape)
        lm_img = lm_wrap.get_landmark_image(im, shape, should_save=True)
        lm_wrap.get_transform_landmarks(im, lm_img, should_save=True)

    write_csv(score, CsvConsts.CSV_VALUES_LABELS, folder, fConsts.MY_VALIDATION_CSV, print_data=True)
    # test read csv
    s = read_csv(folder, fConsts.MY_VALIDATION_CSV, print_data=True)
    print(s)
    images.append('.csv')
    images.append('\\image_03219.pts')
    to_remove = get_files_list(folder, exclude_strings=images)
    print(to_remove)
    for remove_file in to_remove:
        os.remove(remove_file)
