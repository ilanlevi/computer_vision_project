import time
from multiprocessing import Pool
from random import randint

import numpy as np

from consts.datasets_consts import DataFileConsts as dConsts
from consts.fpn_model_consts import FPNConsts
from data.labeled_data import LabeledData
from models.fpn_wrapper import load_fpn_model, get_3d_pose
from utils.image_tools import save_images

"""
    This is a generating for checking validation set scores
"""


def align_images(cam_m, model_m, dataset, prefix_string=''):
    scores_vectors = []
    image_list = []

    for i in range(len(dataset.original_file_list)):
        lmarks = dataset.y_train_set[i]
        img = dataset.x_train_set[i]

        splits = dataset.original_file_list[i].split('\\')
        name = prefix_string + '___' + splits[-1]

        rx, ry, rz, tx, ty, tz = get_3d_pose(cam_m, model_m, lmarks)

        # save: (pitch, yaw, roll, tx, ty, tz)
        this_score = [rx, ry, rz, tx, ty, tz]

        image_list.append((name, img, lmarks))
        scores_vectors.append(name, this_score)

    return scores_vectors, image_list


def calc_and_write_scores(folder_path, scores_vec, image_list):
    for img_name, data_row in scores_vec:
        # gen name
        # save: (pitch, yaw, roll, tx, ty, tz)
        np.savetxt(folder_path + img_name + '.pose', data_row, fmt='%.4f', delimiter=', ',
                   header='pitch, yaw, roll, tx, ty, tz')
        # filename = get_prefix(data_row[1])
        # if filename != '':
        #     write_csv([data_row], CsvConsts.CSV_LABELS, folder_path, filename + '.csv')

    save_images(image_list, folder_path)

    # todo add aug
    # todo check error
    # todo check genewration_name
    # todo - add run from console


def generate_images(data_type):
    prefix_name = randint(0, 200)

    folder_name = data_type['FOLDER']
    suffix = data_type['IMAGE_SUFFIX']

    print '> Generating images for: ' + folder_name

    ds = LabeledData(data_path=folder + folder_name + '\\', picture_suffix=suffix, image_size=500).init()
    ds.read_data_set()

    scores_vector, images_list = align_images(camera_matrix, model_matrix, ds, str(prefix_name))
    calc_and_write_scores(output_path, scores_vector, images_list)
    return len(images_list)


# load model
model_path = FPNConsts.THIS_PATH + FPNConsts.MODELS_DIR
model_file_name = FPNConsts.POSE_P
model_name = FPNConsts.MODEL_NAME

camera_matrix, model_matrix = load_fpn_model(model_path, model_file_name, model_name)

output_file = dConsts.OUTPUT_FILE_NAME

data_sets_list = dConsts.ALL_DATASETS

folder = dConsts.DOWNLOAD_FOLDER
output_folder = dConsts.OUTPUT_FOLDER
output_path = folder + output_folder

# this will generate a test set

if __name__ == '__main__':
    start_time = time.time()

    pool = Pool(processes=5)  # start 4 worker processes
    # pool.imap_unordered(generate_images, (data_set), )
    pool.map(generate_images, data_sets_list)

    print '> Write images completed! (in: %.2f seconds)' % (time.time() - start_time)
