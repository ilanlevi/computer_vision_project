import os

import cv2
import numpy as np
from keras_preprocessing.image.affine_transformations import apply_affine_transform

from consts import DataSetConsts as dsConsts, ValidationFileConsts as fConsts, CsvConsts
from image_utils import load_image, resize_image_and_landmarks
from my_models import LandmarkWrapper, FpnWrapper
from mytools import get_files_list, write_csv, read_csv, get_projection_matrix, my_flip

"""Please ignore! This will be used for testing LandmarkWrapper, FpnWrapper classes"""

if __name__ == '__main__':
    folder = 'C:\\Work\\ComputerVision\\Project\\tmp\\'

    suffixes = dsConsts.PICTURE_SUFFIX
    images = get_files_list(folder, suffixes, [dsConsts.LANDMARKS_FILE_SUFFIX, dsConsts.LANDMARKS_PREFIX])

    lm_wrap = LandmarkWrapper(save_to_dir=True, out_image_size=250)
    fpn_wrap = FpnWrapper()

    score = []
    for im in images:
        lmarks = lm_wrap.load_image_landmarks(im)
        rx, ry, rz, tx, ty, tz = fpn_wrap.get_3d_vectors(lmarks)
        score.append([rx, ry, rz, tx, ty, tz])

        img = load_image(im)
        img, lmarks = resize_image_and_landmarks(img, lmarks, 250)

        lm_img = lm_wrap.get_landmark_image_from_landmarks(lmarks, 250)
        # לבדוק את הלנדמרק מול שלו
        cv2.imshow('image before', img)
        cv2.imshow('points before', lm_img)

        image_shape = (250, 250, 1,)
        lmarks_shpae = (1, 1, 1,)
        # 3 השוואות -> מקורי מול מקורי (שלו), POSE ללנדמרק אחרי הפיכה והשוואה מול POSE אחרי טרנפורמציה
        theta = 1
        img = np.reshape(img, image_shape)
        new_img = apply_affine_transform(img, theta=theta)
        new_img = my_flip(new_img, 0)
        transform_matrix = get_projection_matrix(250, 250, theta=theta)

        new_lmarks = LandmarkWrapper.apply_matrix_to_landmarks(transform_matrix, lmarks)
        new_lmarks = my_flip(new_lmarks, 0)
        new_lmarks_image = lm_wrap.get_landmark_image_from_landmarks(new_lmarks, 250)

        cv2.imshow('image after', new_img)
        cv2.imshow('points after', new_lmarks_image)
        cv2.waitKey(0)

        old_pose = (rx, ry, rz, tx, ty, tz)
        print("old pose: " + str(old_pose))

        new_pose_only_transform = fpn_wrap.apply_transformation_matrix_on_pose(old_pose, transform_matrix,
                                                                               flip_horizontal=True)

        new_img = np.reshape(new_img, (250, 250))
        new_lmarks = np.asarray(new_lmarks)
        new_pose_on_new_lmarks = fpn_wrap.get_3d_vectors(new_lmarks)

        print("new_pose_only_transform: " + str(new_pose_only_transform))
        print("new_pose_on_new_lmarks: " + str(new_pose_on_new_lmarks))

    write_csv(score, CsvConsts.CSV_VALUES_LABELS, folder, fConsts.MY_VALIDATION_CSV)
    # test read csv
    s = read_csv(folder, fConsts.MY_VALIDATION_CSV, print_data=True)
    print(s)
    images.append('.csv')
    images.append('\\image_03219.pts')
    to_remove = get_files_list(folder, exclude_strings=images)
    print(to_remove)
    for remove_file in to_remove:
        os.remove(remove_file)
