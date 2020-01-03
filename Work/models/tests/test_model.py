import cv2
import numpy as np

import Work.models.ThreeD_Model
from Work.consts.csv_consts import CsvConsts
from Work.consts.files_consts import FileConsts as fConsts
from Work.consts.fpn_model_consts import FPNConsts
from Work.data.helen_data import HelenDataSet
from Work.utils import camera_calibration as calib
from Work.utils.dlib_landmarks import DlibLandmarks
from Work.tools.csv_files_tools import write_csv


def generate_dataset():
    ds = HelenDataSet(data_path=fConsts.VALIDATION_FOLDER, label_file_name=fConsts.VALIDATION_CSV, to_gray=True,
                      target_sub=fConsts.PROCESSED_SET_FOLDER, picture_suffix='png')
    ds.init()
    ds.read_data_set()
    return ds


NUMBER_OF_TESTS = 1


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

        # Looping over the faces
        for j in range(len(lmarks)):
            lmark = lmarks[j]
            proj_matrix, camera_matrix, rmat, tvec, rvec = calib.estimate_camera(model3D, lmark)

            print 't_vec' + str(np.asarray(tvec).T)
            print 'r_vec' + str(np.asarray(rvec).T)
            print 'distance' + str(np.asarray(proj_matrix[:, 3]).T)
            _, split = ds.original_file_list[i].split('\\', 1)
            if j > 0:
                score.append([i, split + '(' + str(j) + ')', rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]])
            else:
                score.append([i, split, rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]])

    return score


if __name__ == '__main__':
    s = test_align_image()
    write_csv(s, CsvConsts.CSV_LABELS, fConsts.VALIDATION_FOLDER, fConsts.MY_VALIDATION_CSV, True)

