from math import cos, atan2, asin, sqrt

import numpy as np


# todo delete

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
