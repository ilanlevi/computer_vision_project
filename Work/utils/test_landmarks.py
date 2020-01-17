import cv2

from consts.validation_files_consts import ValidationFileConsts as fConsts
from data.labeled_data import LabeledData
from utils.draw_tools import display_landmarks
from consts.datasets_consts import DataFileConsts as dConsts

"""
    This is a testing class for checking validation set scores
"""


def align_images(data_set, indexes, prefix=''):
    for i in indexes:
        lmarks = data_set.y_train_set[i]
        splits = data_set.original_file_list[i].split('\\')
        name = splits[-1]

        img = data_set.x_train_set[i]
        img = display_landmarks(img, lmarks)
        cv2.imshow("Output - %s - %s" % (name, prefix), img)


if __name__ == '__main__':
    folder = dConsts.DOWNLOAD_FOLDER

    folder_name = dConsts.AFW_DATA['FOLDER']
    suffix = dConsts.AFW_DATA['IMAGE_SUFFIX']

    ds = LabeledData(data_path=folder + folder_name + '\\', picture_suffix=suffix, image_size=500).init() \
        .read_data_set()
    ds2 = LabeledData(data_path=folder + folder_name + '\\', picture_suffix=suffix).init().read_data_set()

    im_idx = [5]

    align_images(ds, im_idx)
    align_images(ds2, im_idx, '2')

    cv2.waitKey(0)
