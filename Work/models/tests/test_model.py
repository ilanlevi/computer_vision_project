import cv2
import numpy as np

import matplotlib.pyplot as plt

import Work.models.ThreeD_Model
from Work.consts.csv_consts import CsvConsts
from Work.consts.files_consts import FileConsts as fConsts
from Work.consts.fpn_model_consts import FPNConsts
from Work.data.helen_data import HelenDataSet
from Work.utils import camera_calibration as calib
from Work.utils.dlib_landmarks import DlibLandmarks
from Work.tools.csv_files_tools import write_csv, read_csv


def generate_dataset():
    ds = HelenDataSet(data_path=fConsts.VALIDATION_FOLDER, label_file_name=fConsts.VALIDATION_CSV, to_gray=True,
                      target_sub=fConsts.PROCESSED_SET_FOLDER, picture_suffix='png')
    ds.init()
    ds.read_data_set()
    return ds


def test_align_image():
    ds = generate_dataset()
    original_images = ds.original_file_list
    pr = DlibLandmarks(fConsts.DOWNLOAD_FOLDER + fConsts.PREDICTOR_FILE_NAME)

    model3D = Work.models.ThreeD_Model.FaceModel(FPNConsts.THIS_PATH + FPNConsts.MODELS_DIR + FPNConsts.POSE_P,
                                                 'model3D',
                                                 False)
    allModels = dict()
    allModels[FPNConsts.POSE] = model3D

    score = []

    for i in range(len(original_images)):
        im = ds.x_train_set[i]
        print ("The Original #%d - %s" % (i, ds.original_file_list[i]))

        lmarks = pr.get_landmarks(im)

        if len(lmarks) == 0:
            print 'No faces in image!'
        # else:
        #     img, lmarks, yaw = calib.flip_in_case(im, lmarks, allModels)

        # Looping over the faces
        for j in range(len(lmarks)):
            lmark = lmarks[j]
            proj_matrix, camera_matrix, rmat, tvec, rvec = calib.estimate_camera(model3D, lmark)

            print 't_vec' + str(np.asarray(tvec).T)
            print 'r_vec' + str(np.asarray(rvec).T)
            print 'distance' + str(np.asarray(proj_matrix[:, 3]).T)
            splits = ds.original_file_list[i].split('\\')
            name = splits[-1]
            if j > 0:
                name = name + '(' + str(j) + ')'
            # score.append([str(i), splits[-1], str(rvec[0][0]), str(rvec[1][0]), str(rvec[2][0]),
            #               str(tvec[0][0]), str(tvec[1][0]), str(tvec[2][0])])
            score.append([i, name, rvec[0][0], rvec[1][0], rvec[2][0],
                          tvec[0][0], tvec[1][0], tvec[2][0], 0])

    return score


def write_scores():
    s = test_align_image()
    write_csv(s, CsvConsts.CSV_LABELS, fConsts.VALIDATION_FOLDER, fConsts.MY_VALIDATION_CSV, True)


def compare_scores():
    my = read_csv(fConsts.VALIDATION_FOLDER, fConsts.MY_VALIDATION_CSV, True)
    valid = read_csv(fConsts.VALIDATION_FOLDER, fConsts.VALIDATION_CSV, True)

    total = []
    for m in my:
        for v in valid:
            if v[CsvConsts.PICTURE_NAME] in m[CsvConsts.PICTURE_NAME]:
                r_x = float(v[CsvConsts.R_X]) - float(m[CsvConsts.R_X])
                r_y = float(v[CsvConsts.R_Y]) - float(m[CsvConsts.R_Y])
                r_z = float(v[CsvConsts.R_Z]) - float(m[CsvConsts.R_Z])
                t_x = float(v[CsvConsts.T_X]) - float(m[CsvConsts.T_X])
                t_y = float(v[CsvConsts.T_Y]) - float(m[CsvConsts.T_Y])
                t_z = float(v[CsvConsts.T_Z]) - float(m[CsvConsts.T_Z])
                total.append([m[CsvConsts.COL_INDEX], m[CsvConsts.PICTURE_NAME], r_x, r_y, r_z, t_x, t_y, t_z, 0])

    write_csv(total, CsvConsts.CSV_LABELS, fConsts.VALIDATION_FOLDER, fConsts.VALIDATION_DIFF_CSV, True)


def plot_diff():
    diff = read_csv(fConsts.VALIDATION_FOLDER, fConsts.VALIDATION_DIFF_CSV, True)
    x_s = range(len(diff))

    R_X = [float(i[CsvConsts.R_X]) for i in diff]
    R_Y = [float(i[CsvConsts.R_Y]) for i in diff]
    R_Z = [float(i[CsvConsts.R_Z]) for i in diff]
    T_X = [float(i[CsvConsts.T_X]) for i in diff]
    T_Y = [float(i[CsvConsts.T_Y]) for i in diff]
    T_Z = [float(i[CsvConsts.T_Z]) for i in diff]

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(x_s, R_X, label=CsvConsts.R_X, alpha=0.5)
    ax1.plot(x_s, R_Y, label=CsvConsts.R_Y, alpha=0.5)
    ax1.plot(x_s, R_Z, label=CsvConsts.R_Z, alpha=0.5)
    ax1.grid(True)

    ax2.plot(x_s, T_X, label=CsvConsts.T_X, alpha=0.5)
    ax2.plot(x_s, T_Y, label=CsvConsts.T_Y, alpha=0.5)
    ax2.plot(x_s, T_Z, label=CsvConsts.T_Z, alpha=0.5)
    ax2.grid(True)
    plt.show()


if __name__ == '__main__':
    # write_scores()
    # compare_scores()
    plot_diff()
