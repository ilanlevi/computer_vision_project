__author__ = 'Douglas and Iacopo'

import numpy as np
import cv2
import math

repLand = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, \
           24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, \
           48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, \
           65, 64, 63, 62, 61, 68, 67, 66]


def flip_in_case(img, lmarks, allModels):
    ## Check if we need to flip the image
    yaws = []  # np.zeros(1,len(allModels))
    ## Getting yaw estimate over poses and subjects
    for mmm in allModels.itervalues():
        proj_matrix, camera_matrix, rmat, tvec, rvecs = estimate_camera(mmm, lmarks[0])
        yaws.append(get_yaw(rmat))
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


def estimate_camera(model3D, fidu_XY, pose_db_on=False):
    if pose_db_on:
        rmat, tvec, ret, rvecs = calib_camera(model3D, fidu_XY)
        tvec = tvec.reshape(3, 1)
    else:
        rmat, tvec, ret, rvecs = calib_camera(model3D, fidu_XY)
    RT = np.hstack((rmat, tvec))
    projection_matrix = model3D.out_A * RT

    return projection_matrix, model3D.out_A, rmat, tvec, rvecs


def calib_camera(model3D, fidu_XY):
    # compute pose using refrence 3D points + query 2D point
    ret = []

    good_ind = np.setdiff1d(np.arange(68) + 1, model3D.indbad)
    good_ind = good_ind - 1
    fid_u_XY = fidu_XY[good_ind, :]
    ret, rvecs, tvec = cv2.solvePnP(model3D.model_TD, fid_u_XY, model3D.out_A, None, None, None, False)

    rmat, jacobian = cv2.Rodrigues(rvecs, None)

    inside = calc_inside(model3D.out_A, rmat, tvec, model3D.size_U[1], model3D.size_U[0], model3D.model_TD)
    if inside == 0:
        tvec = -tvec
        t = np.pi
        RRz180 = np.asmatrix([np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]).reshape((3, 3))
        rmat = RRz180 * rmat
    return rmat, tvec, ret, rvecs


def get_yaw(rmat):
    modelview = rmat
    modelview = np.zeros((3, 4))
    modelview[0:3, 0:3] = rmat.transpose()
    modelview = modelview.reshape(12)
    # Code converted from function: getEulerFromRot()                                                                                                                                                                               
    angle_y = -math.asin(modelview[
                             2])  # Calculate Y-axis angle
    C = math.cos(angle_y)
    angle_y = math.degrees(angle_y)

    if np.absolute(
            C) > 0.005:  # Gimball lock?
        trX = modelview[
                  10] / C  # No, so get X-axis angle
        trY = -modelview[6] / C
        angle_x = math.degrees(math.atan2(trY, trX))

        trX = modelview[
                  0] / C  # Get z-axis angle
        trY = - modelview[1] / C
        angle_z = math.degrees(math.atan2(trY, trX))
    else:
        # Gimball lock has occured                                                                                                                                                                                                       
        angle_x = 0
        trX = modelview[5]
        trY = modelview[4]
        angle_z = math.degrees(math.atan2(trY, trX))

    # Adjust to current mesh setting                                                                                                                                                                                                     
    angle_x = 180 - angle_x
    angle_y = angle_y
    angle_z = -angle_z

    out_pitch = angle_x
    out_yaw = angle_y
    out_roll = angle_z

    print 'out_pitch: ' + str(out_pitch)
    print 'out_yaw: ' + str(out_yaw)
    print 'out_roll: ' + str(out_roll)

    return out_yaw


def get_opengl_matrices(camera_matrix, rmat, tvec, width, height):
    projection_matrix = np.asmatrix(np.zeros((4, 4)))
    near_plane = 0.0001
    far_plane = 10000

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    px = camera_matrix[0, 2]
    py = camera_matrix[1, 2]

    projection_matrix[0, 0] = 2.0 * fx / width
    projection_matrix[1, 1] = 2.0 * fy / height
    projection_matrix[0, 2] = 2.0 * (px / width) - 1.0
    projection_matrix[1, 2] = 2.0 * (py / height) - 1.0
    projection_matrix[2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
    projection_matrix[3, 2] = -1
    projection_matrix[2, 3] = -2.0 * far_plane * near_plane / (far_plane - near_plane)

    deg = 180
    t = deg * np.pi / 180.
    RRz = np.asmatrix([np.cos(t), -np.sin(t), 0, np.sin(t), np.cos(t), 0, 0, 0, 1]).reshape((3, 3))
    RRy = np.asmatrix([np.cos(t), 0, np.sin(t), 0, 1, 0, -np.sin(t), 0, np.cos(t)]).reshape((3, 3))
    rmat = RRz * RRy * rmat

    mv = np.asmatrix(np.zeros((4, 4)))
    mv[0:3, 0:3] = rmat
    mv[0, 3] = tvec[0]
    mv[1, 3] = -tvec[1]
    mv[2, 3] = -tvec[2]
    mv[3, 3] = 1.
    return mv, projection_matrix


def extract_frustum(camera_matrix, rmat, tvec, width, height):
    mv, proj = get_opengl_matrices(camera_matrix, rmat, tvec, width, height)
    clip = proj * mv
    frustum = np.asmatrix(np.zeros((6, 4)))
    # /* Extract the numbers for the RIGHT plane */
    frustum[0, :] = clip[3, :] - clip[0, :]
    # /* Normalize the result */
    v = frustum[0, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[0, :] = frustum[0, :] / t

    # /* Extract the numbers for the LEFT plane */
    frustum[1, :] = clip[3, :] + clip[0, :]
    # /* Normalize the result */
    v = frustum[1, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[1, :] = frustum[1, :] / t

    # /* Extract the BOTTOM plane */
    frustum[2, :] = clip[3, :] + clip[1, :]
    # /* Normalize the result */
    v = frustum[2, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[2, :] = frustum[2, :] / t

    # /* Extract the TOP plane */
    frustum[3, :] = clip[3, :] - clip[1, :]
    # /* Normalize the result */
    v = frustum[3, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[3, :] = frustum[3, :] / t

    # /* Extract the FAR plane */
    frustum[4, :] = clip[3, :] - clip[2, :]
    # /* Normalize the result */
    v = frustum[4, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[4, :] = frustum[4, :] / t

    # /* Extract the NEAR plane */
    frustum[5, :] = clip[3, :] + clip[2, :]
    # /* Normalize the result */
    v = frustum[5, :3]
    t = np.sqrt(np.sum(np.multiply(v, v)))
    frustum[5, :] = frustum[5, :] / t
    return frustum


def calc_inside(camera_matrix, rmat, tvec, width, height, obj_points):
    frustum = extract_frustum(camera_matrix, rmat, tvec, width, height)
    inside = 0
    for point in obj_points:
        if (point_in_frustum(point[0], point[1], point[2], frustum) > 0):
            inside += 1
    return inside


def point_in_frustum(x, y, z, frustum):
    for p in range(0, 3):
        if (frustum[p, 0] * x + frustum[p, 1] * y + frustum[p, 2] + z + frustum[p, 3] <= 0):
            return False
    return True
