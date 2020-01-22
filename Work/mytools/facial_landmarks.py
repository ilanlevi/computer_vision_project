import numpy as np

from mytools import get_prefix


def get_landmarks(img_path, landmarks_suffix='.pts', print_data=False):
    """
    Read and return landmarks from file for given image
    :param img_path: the full image path
    :param landmarks_suffix: the landmarks file suffix
    :param print_data: print data or not (default is false)
    :return: facial landmarks as list or None of failed
    """
    prefix = get_prefix(img_path)
    path = prefix + landmarks_suffix
    try:
        with open(path) as f:
            rows = [rows.strip() for rows in f]

        """Use the curly braces to find the start and end of the point data"""
        head = rows.index('{') + 1
        tail = rows.index('}')

        """Select the point data split into coordinates"""
        raw_points = rows[head:tail]
        coords_set = [point.split() for point in raw_points]

        """Convert entries from lists of strings to tuples of floats"""
        points = [tuple([np.float_(point) for point in coords]) for coords in coords_set]

        return points

    except Exception as e:
        if print_data:
            print 'Error: ' + str(e)
    return None


def get_pose(img_path, pose_suffix='.pose', delimiter=', ', print_data=False):
    """
    Read and return pose from file for given image
    :param delimiter: the delimiter of file. Default is ', '
    :param img_path: the full image path
    :param pose_suffix: the pose file suffix
    :param print_data: print data or not (default is false)
    :return: facial 6Dof pose as array or None of failed
    """
    prefix = get_prefix(img_path)
    path = prefix + pose_suffix
    try:
        data = np.loadtxt(path, delimiter=delimiter)
        if print_data:
            'Data shape: %s (loaded from: %s)' % (str(np.shape(data)), path)
        return data
    except Exception as e:
        if print_data:
            print 'Error: ' + str(e)
    return None
