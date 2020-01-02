from random import randint

import cv2
import numpy as np

import ThreeD_Model
from ..consts.ds_consts import DataSetConsts
from ..consts.files_consts import HelenFileConsts as hfc
from ..data.helen_data import HelenDataSet
from ..utils import camera_calibration as calib
from ..utils import renderer
from ..utils.dlib_landmarks import DlibLandmarks
from ..utils.image_tools import ImageTools


def generate_dataset():
    # ds = HelenDataSet(data_path=hfc.DOWNLOAD_FOLDER, original_sub=hfc.DOWNLOAD_SUB_FOLDER,
    #                   target_sub=hfc.PROCESSED_SET_FOLDER)
    ds = HelenDataSet(data_path=hfc.DOWNLOAD_FOLDER2, original_sub=hfc.VALID_SET_SUB_FOLDER,
                      target_sub=hfc.PROCESSED_SET_FOLDER, picture_suffix='png')
    ds.init()
    return ds

NUMBER_OF_TESTS = 1

POSE = "model3D_aug_-00_00_01"
POSE_P = "model3D_aug_-00_00_01.mat"
THIS_PATH = "C:\\Work\\ComputerVision\\Project\\face_specific_augm-master"
MODELS_DIR = "\\models3d_new\\"

def flipInCase(img, lmarks, allModels):
    ## Index to remap landmarks in case we flip an image
    repLand = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, \
               24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, \
               48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, \
               65, 64, 63, 62, 61, 68, 67, 66]

    ## Check if we need to flip the image
    yaws = []  # np.zeros(1,len(allModels))
    ## Getting yaw estimate over poses and subjects
    for mmm in allModels.itervalues():
        proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(mmm, lmarks[0])
        yaws.append(calib.get_yaw(rmat))
    yaws = np.asarray(yaws)
    yaw = yaws.mean()
    print '> Yaw value mean: ', yaw
    if yaw < 0:
        print '> Positive yaw detected, flipping the image'
        img = cv2.flip(img, 1)
        # Flipping X values for landmarks
        lmarks[0][:, 0] = img.shape[1] - lmarks[0][:, 0]
        # Creating flipped landmarks with new indexing
        lmarks3 = np.zeros((1, 68, 2))
        for i in range(len(repLand)):
            lmarks3[0][i, :] = lmarks[0][repLand[i] - 1, :]
        lmarks = lmarks3
    return img, lmarks, yaw

def test_align_image():
    ds = generate_dataset()
    original_images = ds.original_file_list
    pr = DlibLandmarks(hfc.DOWNLOAD_FOLDER + hfc.PREDICTOR_FILE_NAME)

    model3D = ThreeD_Model.FaceModel(THIS_PATH + MODELS_DIR + POSE_P, 'model3D',
                                     False)
    allModels = dict()
    allModels[POSE] = model3D

    for i in range(NUMBER_OF_TESTS):
        rnd_index = randint(0, len(original_images) - 1)
        image_original = ImageTools.load_images([ds.original_file_list[rnd_index]], width=None)
        images = ImageTools.load_images([ds.original_file_list[rnd_index]], DataSetConsts.PICTURE_WIDTH)
        con_images = ImageTools.load_converted_images([ds.original_file_list[rnd_index]],
                                                      DataSetConsts.PICTURE_WIDTH)

        cv2.imshow(("The Original #%d" % i), image_original[0])

        im = con_images[0]
        img_display = im.copy()
        lmarks = pr.get_landmarks(im)

        im, lmarks, yaw = flipInCase(im, lmarks, allModels)
        listPose = [0, 1, 2, 3, 4]

        ## Looping over the poses
        for poseId in listPose:
            eyemask = model3D.eyemask
            # perform camera calibration according to the first face detected
            proj_matrix, camera_matrix, rmat, tvec = calib.estimate_camera(model3D, lmarks[0])
            print 'proj_matrix' + str(proj_matrix)
            print 'camera_matrix' + str(camera_matrix)
            print 'rmat' + str(rmat)
            print 'tvec' + str(tvec)
            ## We use eyemask only for frontal
            ##### Main part of the code: doing the rendering #############
            rendered_raw, rendered_sym, face_proj, background_proj, temp_proj2_out_2, sym_weight = renderer.render(
                im, proj_matrix, model3D.ref_U, eyemask, model3D.facemask, 10)
            print 'rendered_raw' + str(rendered_raw)
            print 'rendered_sym' + str(rendered_sym)
            print 'face_proj' + str(face_proj)
            print 'background_proj' + str(background_proj)
            print 'temp_proj2_out_2' + str(temp_proj2_out_2)
            print 'sym_weight' + str(sym_weight)
            ########################################################
    print 'bla'


if __name__ == '__main__':
    print 'bla'
    # test_align_image()
