import cv2

from consts.datasets_consts import DataFileConsts as dConsts
from data.labeled_data import LabeledData

# from image_utils.draw_tools import display_landmarks

"""
    This is a testing class for checking validation set scores
"""


# todo remove
def align_images(data_set, indexes, prefix=''):
    for i in indexes:
        lmarks = data_set.y_train_set[i]
        splits = data_set.original_file_list[i].split('\\')
        name = splits[-1]

        img = data_set.x_train_set[i]
        # img = display_landmarks(img, lmarks)
        cv2.imshow("Output - %s - %s" % (name, prefix), img)


if __name__ == '__main__':
    folder = dConsts.DOWNLOAD_FOLDER

    folder_name = dConsts.OUTPUT_FOLDER
    suffixes = [dConsts.AFW_DATA['IMAGE_SUFFIX'], dConsts.W300_DATA['IMAGE_SUFFIX'], dConsts.IBUG_DATA['IMAGE_SUFFIX']]

    ds = LabeledData(data_path=folder + folder_name + '\\', picture_suffix=suffixes).init().read_data_set()

    # todo change
    im_idx = range(50, 55)

    align_images(ds, im_idx)

    cv2.waitKey(0)
