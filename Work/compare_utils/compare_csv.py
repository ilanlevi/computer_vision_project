import cv2
import numpy as np

from consts.csv_consts import CsvConsts
from mytools.csv_files_tools import write_csv, read_csv


def calc_theta_score(A, B):
    """
    Calculate the difference between to images 6DoF:
        theta = np.rad2deg(np.arccos( ( np.trace(A.T @ B) - 1 )/ 2))

    :param A: rotation matrix #1
    :param B: rotation matrix #2
    :return: theta score
    """
    rot_v1 = np.asarray([float(A[CsvConsts.R_X]), float(A[CsvConsts.R_Y]), float(A[CsvConsts.R_Z])])
    rot_v2 = np.asarray([float(B[CsvConsts.R_X]), float(B[CsvConsts.R_Y]), float(B[CsvConsts.R_Z])])

    rot_m_1, _ = cv2.Rodrigues(rot_v1)
    rot_m_2, _ = cv2.Rodrigues(rot_v2)

    r_matrix = rot_m_2.dot(rot_m_1.T)
    angle = (np.trace(r_matrix) - 1) / 2

    # check possible NaN
    if abs(angle) > 1:
        sign = angle / abs(angle)
        angle = 1.0 * sign

    theta = np.arccos(angle)

    theta = np.rad2deg(theta)

    return theta


def print_param_details(param_name, param_index, arr):
    """
    print the result differences for a param
    :param param_name: the param name to print results
    :param param_index: the param index
    :param arr: the difference array
    :return: None - print data
    """
    print '#####'
    print 'avg_' + param_name + ' = ' + str(np.mean(arr[:, param_index]))
    print 'max_' + param_name + ' = ' + str(np.max(arr[:, param_index]))
    print 'min_' + param_name + ' = ' + str(np.min(arr[:, param_index]))
    print '#####'


def compare_scores(folder, f1, f2, f_new, print_data=False):
    """
    Compare scores of 2 csv files 6Dof scores and write the difference in new csv file
    :param folder: the directory path
    :param f1: csv file #1 name
    :param f2: csv file #2 name
    :param f_new: new difference csv file name
    :param print_data: print print_param_details or not (default is False)
    :return: None - save difference in file
    """
    file1_csv = read_csv(folder, f1, True)
    file2_csv = read_csv(folder, f2, True)

    total = []
    total_n = []
    for f2_row in file2_csv:
        for f1_row in file1_csv:
            if f1_row[CsvConsts.PICTURE_NAME] == f2_row[CsvConsts.PICTURE_NAME]:
                r_x = float(f1_row[CsvConsts.R_X]) - float(f2_row[CsvConsts.R_X])
                r_y = float(f1_row[CsvConsts.R_Y]) - float(f2_row[CsvConsts.R_Y])
                r_z = float(f1_row[CsvConsts.R_Z]) - float(f2_row[CsvConsts.R_Z])
                t_x = float(f1_row[CsvConsts.T_X]) - float(f2_row[CsvConsts.T_X])
                t_y = float(f1_row[CsvConsts.T_Y]) - float(f2_row[CsvConsts.T_Y])
                t_z = float(f1_row[CsvConsts.T_Z]) - float(f2_row[CsvConsts.T_Z])
                theta = calc_theta_score(f2_row, f1_row)

                total.append(
                    [f2_row[CsvConsts.COL_INDEX], f2_row[CsvConsts.PICTURE_NAME], r_x, r_y, r_z, t_x, t_y, t_z, theta])
                total_n.append([r_x, r_y, r_z, t_x, t_y, t_z, theta])

    total_arr = np.asarray(total_n, dtype=np.float)
    total_arr = np.reshape(total_arr, (len(total), 7))

    if print_data:
        for param_index in range(len(CsvConsts.CSV_VALUES_LABELS)):
            print_param_details(CsvConsts.CSV_VALUES_LABELS[param_index], param_index, total_arr)

        print_param_details('theta', 6, total_arr)

    write_csv(total, CsvConsts.CSV_LABELS_DIFF, folder, f_new, True)
