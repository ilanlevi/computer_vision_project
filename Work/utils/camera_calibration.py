__author__ = 'Douglas and Iacopo'

import cv2
import numpy as np


def estimate_camera(model3D, fidu_XY):
    rmat, t, r_exp = calib_camera(model3D, fidu_XY)
    RT = np.hstack((rmat, t))
    projection_matrix = model3D.out_A * RT

    return projection_matrix, model3D.out_A, rmat, t, r_exp


def calib_camera(model3D, fidu_XY):
    # compute pose using refrence 3D points + query 2D point
    ret = []

    good_ind = np.setdiff1d(np.arange(68) + 1, model3D.indbad)
    good_ind = good_ind - 1
    fid_u_XY = fidu_XY[good_ind, :]
    _, r_exp, t = cv2.solvePnP(model3D.model_TD, fid_u_XY, model3D.out_A, None, None, None, False)
    # ret, r_exp, tvec = cv2.solvePnP(model3D.model_TD, fid_u_XY, model3D.out_A, None, None, None, False)

    # rmat, _ = cv2.Rodrigues(r_exp, None)
    rmat, _ = cv2.Rodrigues(r_exp, None)
    #
    # inside = calc_inside(model3D.out_A, rmat, tvec, model3D.size_U[1], model3D.size_U[0], model3D.model_TD)
    # if inside == 0:
    #     tvec = -tvec
    #     t = np.pi
    #     RRz180 = np.asmatrix([np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]).reshape((3, 3))
    #     rmat = RRz180 * rmat
    return rmat, t, r_exp
