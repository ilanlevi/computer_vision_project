import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np

import Work.models.ThreeD_Model
from Work.consts.csv_consts import CsvConsts
from Work.consts.files_consts import FileConsts as fConsts
from Work.consts.fpn_model_consts import FPNConsts
from Work.data.helen_data import HelenDataSet
from Work.tools.csv_files_tools import write_csv, read_csv
from Work.utils import camera_calibration as calib
from Work.utils.facial_landmarks import FacialLandmarks

valid_csv = read_csv(fConsts.VALIDATION_FOLDER, fConsts.VALIDATION_CSV, False)


def generate_dataset():
    ds = HelenDataSet(data_path=fConsts.VALIDATION_FOLDER, label_file_name=fConsts.VALIDATION_CSV, to_gray=True,
                      target_sub=fConsts.PROCESSED_SET_FOLDER, picture_suffix='png')
    ds.init()
    ds.read_data_set()
    return ds


def test_align_image():
    ds = generate_dataset()
    original_images = ds.original_file_list

    model3D = Work.models.ThreeD_Model.FaceModel(FPNConsts.THIS_PATH + FPNConsts.MODELS_DIR + FPNConsts.POSE_P,
                                                 'model3D',
                                                 True)
    allModels = dict()
    allModels[FPNConsts.POSE] = model3D

    score_vectors = []

    # mean_img = np.mean(ds.x_train_set)
    total_dff = 0

    for i in range(len(original_images)):
        im = ds.x_train_set[i]
        # im = im - mean_img
        # print ("The Original #%d - %s" % (i, ds.original_file_list[i]))

        # lmarks = pr.get_landmarks(im)
        lmarks = FacialLandmarks.get_landmarks(ds.original_file_list[i])
        # if len(lmarks) == 0:
        #     print 'No faces in image!'
        if len(lmarks) > 0:
            # lmarks = np.asarray(lmarks)
            shape = np.shape(lmarks)
            lmarks = np.reshape(lmarks, (shape[0], shape[1]))
            # img, lmarks, yaw = calib.flip_in_case(im, lmarks, allModels)

            # ret, rvec, tvec = cv2.solvePnP(model3D.model_TD, lmarks, model3D.out_A, None, None, None, False)

            splits = ds.original_file_list[i].split('\\')
            name = splits[-1]
            projection_matrix, model3D.out_A, rmat, t, r_exp = calib.estimate_camera(model3D, lmarks)
            r_vect, _ = cv2.Rodrigues(rmat)
            r_vect = np.squeeze(r_vect)
            for q in range(len(r_vect)):
                r_vect[q] = np.math.radians(np.math.degrees(r_vect[q]))
            # out_im = lmark * proj_matrix
            # eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
            rx, ry, rz = r_vect[0], r_vect[1], r_vect[2]

            # rx, ry, rz = [math.radians(_) for _ in eulerAngles]
            # rx = math.degrees(math.asin(math.sin(rx)))
            # ry = math.degrees(math.asin(math.sin(ry)))
            # rz = math.degrees(math.asin(math.sin(rz)))
            #
            # rx, ry, rz = [math.radians(_) for _ in [rx, ry, rz]]

            # rx = rx + (rmat2[2][1] + rmat2[1][2])
            # ry = ry + (rmat2[2][0] + rmat2[0][2])
            # rz = rz + (rmat2[1][0] + rmat2[0][1])
            # ImagesPose.P2sRt(proj_matrix)
            # scale, rotation_m, proj_m = ImagesPose.P2sRt(proj_matrix)
            # _, _, rz = rotationMatrixToEulerAngles(rmat)
            # tx, ty, tz = tvec[0][0], tvec[1][0], tvec[2][0]
            tx, ty, tz = t[0][0], t[1][0], t[2][0]
            score_vectors.append([i, name, rx, ry, rz, tx, ty, tz, 0])
            # valid = [item for item in valid_csv if item[CsvConsts.PICTURE_NAME] in name]
            # if len(valid) > 0:
            #     valid = valid[0]
            # if len(valid) > 0 and (np.abs(rx - np.float(valid[CsvConsts.R_X])) > 0.2
            #                        or np.abs(ry - np.float(valid[CsvConsts.R_Y])) > 0.2
            #                        or np.abs(rz - np.float(valid[CsvConsts.R_Z])) > 0.2):
            #     # print name
            #     # cv2.imshow(name, im)
            #     total_dff = total_dff + 1

    # print ' >> total diff = %d' % total_dff

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

    # r, _ = cv2.Rodrigues(rot_m_2.dot(rot_m_1.T))
    # rotation_error_from_identity = np.linalg.norm(r)
    # r, _ = cv2.Rodrigues(rot_m_2.dot(rot_m_1.T))
    # rotation_error_from_identity = np.linalg.norm(r)
    # r_matrix = np.dot(rot_m_1.T, rot_m_2)
    # r_matrix = np.dot(rot_m_1, rot_m_2)
    # theta = np.angle([rot_m_1.T, rot_m_2], True)
    # r_matrix = (rot_m_1.T * rot_m_2)
    r_matrix = rot_m_2.dot(rot_m_1.T)
    angle = (np.trace(r_matrix) - 1) / 2
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

    # print_param_details('rx', 0, total_arr)
    # print_param_details('ry', 1, total_arr)
    # print_param_details('rz', 2, total_arr)
    # print_param_details('tx', 3, total_arr)
    # print_param_details('ty', 4, total_arr)
    # print_param_details('tz', 5, total_arr)
    print_param_details('theta', 6, total_arr)

    write_csv(total, CsvConsts.CSV_LABELS, folder, f_new, True)


def plt_axs(axs, x, y, label, labels):
    axs.plot(x, y, label=label, alpha=0.5)
    # zip joins x and y coordinates in pairs
    diff = []
    for x_s in x:
        if abs(y[x_s]) > 0.05:
            diff.append(labels[x_s])
    if len(diff) > 0:
        print '> %s - (total %d) diff in %s ' % (label, len(diff), str(diff))


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
    # plot_diff(folder=fConsts.VALIDATION_FOLDER, filename=fConsts.MY_VALIDATION_CSV, title='my',
    #           print_scores=False)
    # plot_diff(folder=fConsts.VALIDATION_FOLDER, filename=fConsts.VALIDATION_CSV, title='validation',
    #           print_scores=False)
    compare_scores(folder=fConsts.VALIDATION_FOLDER, f1=fConsts.VALIDATION_CSV, f2=fConsts.MY_VALIDATION_CSV,
                   f_new=fConsts.VALIDATION_DIFF_CSV)
    plot_diff(folder=fConsts.VALIDATION_FOLDER, filename=fConsts.VALIDATION_DIFF_CSV, title='diff',
              print_scores=False)

    plt.show()
