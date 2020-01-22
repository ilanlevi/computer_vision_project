from random import randint

import matplotlib.pyplot as plt
import cv2
# from image_utils.draw_tools import roi_from_landmarks, display_landmarks
from compare_utils import plot_diff, plot_diff_each_param, compare_scores
from consts.csv_consts import CsvConsts
from consts.fpn_model_consts import FPNConsts
from consts.validation_files_consts import ValidationFileConsts as fConsts
from data.labeled_data import LabeledData
from models.fpn_wrapper import load_fpn_model, get_3d_pose
from mytools.csv_files_tools import write_csv, read_csv

"""
    This is a testing class for checking validation set scores
"""


def generate_dataset(files_list):
    # create dataset
    ds = LabeledData(data_path=fConsts.VALIDATION_FOLDER, to_gray=True, picture_suffix='png').init()

    files_list = [(fConsts.VALIDATION_FOLDER + fConsts.VALID_SET_SUB_FOLDER + file_name) for file_name in files_list]

    ds.original_file_list = files_list
    ds.read_data_set()

    # return dataset with files that appear in validation csv only
    return ds


def load_data_and_models():
    # load validation dataset
    original_set = read_csv(fConsts.VALIDATION_FOLDER, fConsts.VALIDATION_CSV)
    file_list = [validation_csv[CsvConsts.PICTURE_NAME] for validation_csv in original_set]
    data_set = generate_dataset(file_list)

    # load model
    model_path = FPNConsts.THIS_PATH + FPNConsts.MODELS_DIR
    model_file_name = FPNConsts.POSE_P
    model_name = FPNConsts.MODEL_NAME

    camera_matrix, model_matrix = load_fpn_model(model_path, model_file_name, model_name)

    return camera_matrix, model_matrix, data_set


def align_images(camera_matrix, model_matrix, data_set):
    scores_vectors = []
    # rnd = randint(0, len(data_set.original_file_list) - 1)
    for i in range(len(data_set.original_file_list)):
        lmarks = data_set.y_train_set[i]

        splits = data_set.original_file_list[i].split('\\')
        name = splits[-1]

        rx, ry, rz, tx, ty, tz = get_3d_pose(camera_matrix, model_matrix, lmarks)
        this_score = [i, name, rx, ry, rz, tx, ty, tz]

        # if i == rnd:
        #     img = data_set.x_train_set[i]
        #     img = display_landmarks(img, lmarks)
        #     roi = roi_from_landmarks(img, lmarks)
        #     cv2.imshow("Output - %s" % name, roi)

        scores_vectors.append(this_score)

    return scores_vectors


def write_scores(folder_path, filename, camera_matrix, model_matrix, data_set, print_scores=False):
    s = align_images(camera_matrix, model_matrix, data_set)
    write_csv(s, CsvConsts.CSV_LABELS, folder_path, filename, print_scores)


if __name__ == '__main__':
    cam_m, model_m, data_s = load_data_and_models()

    folder = fConsts.VALIDATION_FOLDER
    filename_my = fConsts.MY_VALIDATION_CSV
    filename_valid = fConsts.VALIDATION_CSV
    filename_diff = fConsts.VALIDATION_DIFF_CSV

    write_scores(folder, filename_my, cam_m, model_m, data_s)

    compare_scores(folder, filename_valid, filename_my, filename_diff, False)
    plot_diff(folder, filename_diff, title='diff')

    plot_diff_each_param(folder, [filename_valid, filename_my])

    plt.show()
