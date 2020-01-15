import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np

import Work.models.ThreeD_Model
from Work.consts.csv_consts import CsvConsts
from Work.consts.fpn_model_consts import FPNConsts
from Work.consts.validation_files_consts import ValidationFileConsts as fConsts
from Work.data.labeled_data import LabeledData
from Work.mytools.csv_files_tools import write_csv, read_csv

valid_csv = read_csv(fConsts.VALIDATION_FOLDER, fConsts.VALIDATION_CSV, False)


def generate_dataset(list_f):
    ds = LabeledData(data_path=fConsts.VALIDATION_FOLDER, label_file_name=fConsts.VALIDATION_CSV, to_gray=False,
                     target_sub=fConsts.PROCESSED_SET_FOLDER, picture_suffix='png')
    ds.init()
    list_f = [(fConsts.VALIDATION_FOLDER + fConsts.VALID_SET_SUB_FOLDER + file_name) for file_name in list_f]
    ds.original_file_list = list_f
    ds.read_data_set()
    return ds


def test_align_image():
    original = read_csv(fConsts.VALIDATION_FOLDER, fConsts.VALIDATION_CSV)
    m_list = [im[CsvConsts.PICTURE_NAME] for im in original]
    ds = generate_dataset(m_list)

    model3D = Work.models.ThreeD_Model.FaceModel(FPNConsts.THIS_PATH + FPNConsts.MODELS_DIR + FPNConsts.POSE_P,
                                                 'model3D',
                                                 False)
    allModels = dict()
    allModels[FPNConsts.POSE] = model3D

    score_vectors = []

    for i in range(len(m_list)):
        # image = ds.x_train_set[i]

        lmarks = ds.y_train_set[i]

        splits = ds.original_file_list[i].split('\\')
        name = splits[-1]
        _, r_exp, tvec = cv2.solvePnP(model3D.model_TD, lmarks, model3D.out_A, None, None, None, False)

        pose = np.squeeze(r_exp)
        t3d = np.squeeze(tvec)

        # set pitch, yaw, roll
        rx = pose[0]
        ry = pose[1]
        rz = pose[2]
        # rx, ry, rz = get_yaw(rmat)

        tx = t3d[0]
        ty = t3d[1]
        tz = t3d[2]

        this_score = [i, name, rx, ry, rz, tx, ty, tz]

        score_vectors.append(this_score)

    # cv2.waitKey(0)
    return score_vectors


def write_scores(folder, filename, print_scores=True):
    s = test_align_image()
    write_csv(s, CsvConsts.CSV_LABELS, folder, filename, print_scores)


def print_param_details(param_name, param_index, arr):
    print '#####'
    print 'avg_' + param_name + ' = ' + str(np.mean(arr[:, param_index]))
    print 'max_' + param_name + ' = ' + str(np.max(arr[:, param_index]))
    print 'min_' + param_name + ' = ' + str(np.min(arr[:, param_index]))
    print '#####'


def calc_theta_score(s1, s2):
    rot_v1 = np.asarray([float(s1[CsvConsts.R_X]), float(s1[CsvConsts.R_Y]), float(s1[CsvConsts.R_Z])])
    rot_v2 = np.asarray([float(s2[CsvConsts.R_X]), float(s2[CsvConsts.R_Y]), float(s2[CsvConsts.R_Z])])

    rot_m_1, _ = cv2.Rodrigues(rot_v1)
    rot_m_2, _ = cv2.Rodrigues(rot_v2)

    r_matrix = rot_m_2.dot(rot_m_1.T)
    angle = (np.trace(r_matrix) - 1) / 2

    # check possible NaN
    if abs(angle) > 1:
        if angle < 0:
            angle = -1
        else:
            angle = 1

    theta = np.arccos(angle)

    theta = np.rad2deg(theta)

    return theta


def compare_scores(folder, f1, f2, f_new):
    valid = read_csv(folder, f1, True)
    my = read_csv(folder, f2, True)

    total = []
    total_n = []
    for m in my:
        for v in valid:
            if v[CsvConsts.PICTURE_NAME] == m[CsvConsts.PICTURE_NAME]:
                r_x = float(v[CsvConsts.R_X]) - float(m[CsvConsts.R_X])
                r_y = float(v[CsvConsts.R_Y]) - float(m[CsvConsts.R_Y])
                r_z = float(v[CsvConsts.R_Z]) - float(m[CsvConsts.R_Z])
                t_x = float(v[CsvConsts.T_X]) - float(m[CsvConsts.T_X])
                t_y = float(v[CsvConsts.T_Y]) - float(m[CsvConsts.T_Y])
                t_z = float(v[CsvConsts.T_Z]) - float(m[CsvConsts.T_Z])
                theta = calc_theta_score(m, v)

                total.append([m[CsvConsts.COL_INDEX], m[CsvConsts.PICTURE_NAME], r_x, r_y, r_z, t_x, t_y, t_z, theta])
                total_n.append([r_x, r_y, r_z, t_x, t_y, t_z, theta])

    total_arr = np.asarray(total_n, dtype=np.float)
    total_arr = np.reshape(total_arr, (len(total), 7))

    print_param_details('rx', 0, total_arr)
    print_param_details('ry', 1, total_arr)
    print_param_details('rz', 2, total_arr)
    print_param_details('tx', 3, total_arr)
    print_param_details('ty', 4, total_arr)
    print_param_details('tz', 5, total_arr)
    print_param_details('theta', 6, total_arr)

    write_csv(total, CsvConsts.CSV_LABELS_DIFF, folder, f_new, True)


def plt_axs(axs, x, y, label, labels):
    # y = y[y != 0]
    # zip joins x and y coordinates in pairs
    diff = []
    for x_s in x:
        if abs(y[x_s]) > 0.001:
            diff.append(labels[x_s])
    if len(diff) > 0:
        print '> %s - (total %d) diff in %s ' % (label, len(diff), str(diff))
    bins = range(len(x))
    # axs.hist(bins, y, label=label, density=True, histtype='bar', rwidth=0.8)
    axs.plot(x, y, label=label, alpha=0.5)


def plot_diff_each_param(folder, filename1, filename2, print_scores=True):
    file1 = read_csv(folder, filename1, print_scores)
    file2 = read_csv(folder, filename2, print_scores)

    fields = CsvConsts.CSV_VALUES_LABELS
    f_1_values = [[float(fileRow[field]) for fileRow in file1] for field in fields]
    f_2_values = [[float(fileRow[field]) for fileRow in file2] for field in fields]

    x_s = range(len(file1))

    fig, axs = plt.subplots(len(fields), 1)

    for i in range(len(fields)):
        axs[i].plot(x_s, f_1_values[i], label=(fields[i] + '  ' + filename1), alpha=0.5)
        axs[i].plot(x_s, f_2_values[i], label=(fields[i] + '  ' + filename2), alpha=0.5)
        axs[i].grid(True)
        axs[i].legend()


def plot_diff(folder=fConsts.VALIDATION_FOLDER, filename=fConsts.VALIDATION_DIFF_CSV2, title='', print_scores=True):
    diff = read_csv(folder, filename, print_scores)
    x_s = range(len(diff))

    LABELS = [i[CsvConsts.PICTURE_NAME] for i in diff]
    R_X = [float(i[CsvConsts.R_X]) for i in diff]
    R_Y = [float(i[CsvConsts.R_Y]) for i in diff]
    R_Z = [float(i[CsvConsts.R_Z]) for i in diff]
    T_X = [float(i[CsvConsts.T_X]) for i in diff]
    T_Y = [float(i[CsvConsts.T_Y]) for i in diff]
    T_Z = [float(i[CsvConsts.T_Z]) for i in diff]

    has_theta = diff[0].has_key(CsvConsts.THETA)
    if has_theta:
        THETA = [float(i[CsvConsts.THETA]) for i in diff]
    else:
        THETA = [float(0) for i in diff]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle(title)

    plt_axs(ax1, x_s, R_X, CsvConsts.R_X, LABELS)
    plt_axs(ax1, x_s, R_Y, CsvConsts.R_Y, LABELS)
    plt_axs(ax1, x_s, R_Z, CsvConsts.R_Z, LABELS)
    ax1.grid(True)
    ax1.legend()

    plt_axs(ax2, x_s, T_X, CsvConsts.T_X, LABELS)
    plt_axs(ax2, x_s, T_Y, CsvConsts.T_Y, LABELS)
    plt_axs(ax2, x_s, T_Z, CsvConsts.T_Z, LABELS)
    ax2.grid(True)
    ax2.legend()

    # ax3.plot(x_s, THETA, label=CsvConsts.THETA, alpha=0.5)
    plt_axs(ax3, x_s, THETA, CsvConsts.THETA, LABELS)
    ax3.grid(True)
    ax3.legend()


if __name__ == '__main__':
    write_scores(folder=fConsts.VALIDATION_FOLDER, filename=fConsts.MY_VALIDATION_CSV, print_scores=False)
    compare_scores(folder=fConsts.VALIDATION_FOLDER, f1=fConsts.VALIDATION_CSV, f2=fConsts.MY_VALIDATION_CSV,
                   f_new=fConsts.VALIDATION_DIFF_CSV2)
    plot_diff(folder=fConsts.VALIDATION_FOLDER, filename=fConsts.VALIDATION_DIFF_CSV, title='diff',
              print_scores=False)
    plot_diff_each_param(fConsts.VALIDATION_FOLDER, fConsts.VALIDATION_CSV, fConsts.MY_VALIDATION_CSV)

    plt.show()
