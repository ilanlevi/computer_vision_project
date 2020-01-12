import itertools

import cv2
import matplotlib.pyplot as plt
import numpy as np

import Work.models.ThreeD_Model
from Work.consts.csv_consts import CsvConsts
from Work.consts.fpn_model_consts import FPNConsts
from Work.consts.validation_files_consts import ValidationFileConsts as fConsts
from Work.data.my_data import LabeledData
from Work.mytools.csv_files_tools import write_csv, read_csv
from Work.utils import camera_calibration as calib, matrix2angle, draw_axis_on_image, get_camarx_matrix, \
    display_landmarks

valid_csv = read_csv(fConsts.VALIDATION_FOLDER, fConsts.VALIDATION_CSV, False)


def generate_dataset():
    ds = LabeledData(data_path=fConsts.VALIDATION_FOLDER, label_file_name=fConsts.VALIDATION_CSV, to_gray=True,
                     target_sub=fConsts.PROCESSED_SET_FOLDER, picture_suffix='png')
    ds.init()
    ds.read_data_set()
    return ds


def get_same(original, name):
    for v in original:
        if v[CsvConsts.PICTURE_NAME] in name:
            return v
    return None


def test_align_image2():
    ds = generate_dataset()

    model3D = Work.models.ThreeD_Model.FaceModel(FPNConsts.THIS_PATH + FPNConsts.MODELS_DIR + FPNConsts.POSE_P,
                                                 'model3D',
                                                 False)
    allModels = dict()
    allModels[FPNConsts.POSE] = model3D

    original = read_csv(fConsts.VALIDATION_FOLDER, fConsts.VALIDATION_CSV)
    score_vectors = []

    for i in range(len(ds.y_train_set)):
        lmarks = ds.y_train_set[i]

        if len(lmarks) > 0:

            image = ds.x_train_set[i]
            camera_matrix, roi = get_camarx_matrix(image, lmarks)

            splits = ds.original_file_list[i].split('\\')
            name = splits[-1]

            # dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
            # (success, r_exp, tvec) = cv2.solvePnP(model3D.model_TD, lmarks, camera_matrix,
            #                                       dist_coeffs, flags=cv2.CV_ITERATIVE)

            _, r_exp, tvec = cv2.solvePnP(model3D.model_TD, lmarks, model3D.out_A, None, None, None, False)
            # _, r_exp, tvec = cv2.solvePnP(model3D.model_TD, lmarks, camera_matrix, None, None, None, False)

            # rmat, _ = cv2.Rodrigues(r_exp)
            # pose = matrix2angle(rmat)  # yaw, pitch, roll
            # pose = np.squeeze(pose)
            pose = np.squeeze(r_exp)
            t3d = np.squeeze(tvec)

            # set pitch, yaw, roll
            rx = pose[0]
            ry = pose[1]
            rz = pose[2]

            # set translation_x, translation_y, translation_z
            tx = t3d[0]
            ty = t3d[1]
            tz = t3d[2]

            if i in [25, 34, 35, 40, 41]:
                img = draw_axis_on_image(image, rx, ry, rz, tx, ty, tz, camera_matrix)
                display_landmarks(img, lmarks, name + '   -   ' + str(i))

            score_vectors.append([i, name, rx, ry, rz, tx, ty, tz, 0])

    # cv2.waitKey(0)
    return score_vectors


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

    for i in range(len(ds.x_train_set)):
        # im = im - mean_img
        # print ("The Original #%d - %s" % (i, ds.original_file_list[i]))

        # lmarks = pr.get_landmarks(im)
        lmarks = ds.y_train_set[i]
        # if len(lmarks) == 0:
        #     print 'No faces in image!'
        if len(lmarks) > 0:
            # lmarks = np.asarray(lmarks)
            shape = np.shape(lmarks)
            lmarks = np.reshape(lmarks, (shape[0], shape[1]))
            # img, lmarks, yaw = calib.flip_in_case(im, lmarks, allModels)

            # ret, rvec, tvec = cv20.solvePnP(model3D.model_TD, lmarks, model3D.out_A, None, None, None, False)

            splits = ds.original_file_list[i].split('\\')
            name = splits[-1]
            projection_matrix, bla, rmat, t, r_exp = calib.estimate_camera(model3D, lmarks)
            r_vect, _ = cv2.Rodrigues(rmat)
            r_vect = np.squeeze(r_vect)
            # r_vect = np.squeeze( r_exp)
            #
            # for q in range(len(r_vect)):
            #     r_vect[q] = np.math.radians(np.math.degrees(r_vect[q]))
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
    write_csv(s, CsvConsts.CSV_LABELS_DIFF, folder, filename, print_scores)


def write_scores2(folder, filename, print_scores=True):
    # s = test_align_image()
    s = test_align_image2()
    write_csv(s, CsvConsts.CSV_LABELS_DIFF, folder, filename, print_scores)


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
    if 1 - abs(angle) < 0:
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

    # print_param_details('rx', 0, total_arr)
    # print_param_details('ry', 1, total_arr)
    # print_param_details('rz', 2, total_arr)
    # print_param_details('tx', 3, total_arr)
    # print_param_details('ty', 4, total_arr)
    # print_param_details('tz', 5, total_arr)
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
    write_scores2(folder=fConsts.VALIDATION_FOLDER, filename=fConsts.MY_VALIDATION_CSV2, print_scores=False)
    compare_scores(folder=fConsts.VALIDATION_FOLDER, f1=fConsts.VALIDATION_CSV, f2=fConsts.MY_VALIDATION_CSV,
                   f_new=fConsts.VALIDATION_DIFF_CSV)
    compare_scores(folder=fConsts.VALIDATION_FOLDER, f1=fConsts.VALIDATION_CSV, f2=fConsts.MY_VALIDATION_CSV2,
                   f_new=fConsts.VALIDATION_DIFF_CSV2)
    compare_scores(folder=fConsts.VALIDATION_FOLDER, f1=fConsts.MY_VALIDATION_CSV, f2=fConsts.MY_VALIDATION_CSV2,
                   f_new=fConsts.VALIDATION_DIFF_CSV3)
    plot_diff(folder=fConsts.VALIDATION_FOLDER, filename=fConsts.VALIDATION_DIFF_CSV, title='diff',
              print_scores=False)
    plot_diff(folder=fConsts.VALIDATION_FOLDER, filename=fConsts.VALIDATION_DIFF_CSV2, title='diff - 2',
              print_scores=False)
    plot_diff(folder=fConsts.VALIDATION_FOLDER, filename=fConsts.VALIDATION_DIFF_CSV3, title='diff - 3',
              print_scores=False)

    plt.show()
