from math import cos, sin, atan2, asin, sqrt
import numpy as np
from .my_io import _load
from ..consts import files_consts as fc


class ImagesPose:
    @staticmethod
    def parse_roi_box_from_landmark(pts):
        """calc roi box from landmark"""
        bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
        center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
        bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

        llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
        center_x = (bbox[2] + bbox[0]) / 2
        center_y = (bbox[3] + bbox[1]) / 2

        roi_box = [0] * 4
        roi_box[0] = center_x - llength / 2
        roi_box[1] = center_y - llength / 2
        roi_box[2] = roi_box[0] + llength
        roi_box[3] = roi_box[1] + llength

        return roi_box

    @staticmethod
    def parse_roi_box_from_bbox(bbox):
        left, top, right, bottom = bbox
        old_size = (right - left + bottom - top) / 2
        center_x = right - (right - left) / 2.0
        center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
        size = int(old_size * 1.58)
        roi_box = [0] * 4
        roi_box[0] = center_x - size / 2
        roi_box[1] = center_y - size / 2
        roi_box[2] = roi_box[0] + size
        roi_box[3] = roi_box[1] + size
        return roi_box

    @staticmethod
    def parse_pose(param, meta):
        # param_mean and param_std are used for re-whitening
        param_mean = meta.get('param_mean')
        param_std = meta.get('param_std')
        print 'param shape: ' + str(np.shape(param))
        print 'param_mean shape: ' + str(np.shape(param_mean))
        print 'param_std shape: ' + str(np.shape(param_std))
        param = np.transpose(param[0]) * param_std + param_mean
        p_s = param[:12].reshape(3, -1)  # camera matrix
        # R = P[:, :3]
        s, r, t3d = ImagesPose.P2sRt(p_s)
        p = np.concatenate((r, t3d.reshape(3, -1)), axis=1)  # without scale
        # P = Ps / s
        pose = ImagesPose.matrix2angle(r)  # yaw, pitch, roll
        # offset = p_[:, -1].reshape(3, 1)
        return p, pose

    @staticmethod
    def matrix2angle(R):
        """ compute three Euler angles from a Rotation Matrix. Ref: http://www.gregslabaugh.net/publications/euler.pdf
        Args:
            R: (3,3). rotation matrix
        Returns:
            x: yaw
            y: pitch
            z: roll
        """
        # assert(isRotationMatrix(R))

        if R[2, 0] != 1 and R[2, 0] != -1:
            x = asin(R[2, 0])
            y = atan2(R[2, 1] / cos(x), R[2, 2] / cos(x))
            z = atan2(R[1, 0] / cos(x), R[0, 0] / cos(x))

        else:  # Gimbal lock
            z = 0  # can be anything
            if R[2, 0] == -1:
                x = np.pi / 2
                y = z + atan2(R[0, 1], R[0, 2])
            else:
                x = -np.pi / 2
                y = -z + atan2(-R[0, 1], -R[0, 2])

        return x, y, z

    @staticmethod
    def P2sRt(P):
        """
         decompositing camera matrix P.
        Args:
            P: (3, 4). Affine Camera Matrix.
        Returns:
            s: scale factor.
            R: (3, 3). rotation matrix.
            t2d: (2,). 2d translation.
        """
        t3d = P[:, 3]
        R1 = P[0:1, :3]
        R2 = P[1:2, :3]
        s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
        r1 = R1 / np.linalg.norm(R1)
        r2 = R2 / np.linalg.norm(R2)
        r3 = np.cross(r1, r2)

        R = np.concatenate((r1, r2, r3), 0)
        return s, R, t3d

    def __init__(self):
        pass
