import os
import pickle

import numpy as np


def get_files_list(path, suffixes=None, exclude_strings=None):
    """
        return file list from path with the same
         suffixes (if none all of the returned)
    """

    if suffixes is None:
        suffixes = []

    if not isinstance(suffixes, list):
        suffixes = [suffixes]

    if exclude_strings is not None and not isinstance(exclude_strings, list):
        exclude_strings = [exclude_strings]

    suffixes = np.unique(suffixes)

    files = [os.path.join(r, file_) for r, d, f in os.walk(path) for file_ in f]
    f_list = []
    for file_name in files:
        if len(suffixes) == 0:
            f_list.append(file_name)
        else:
            for suffix in suffixes:
                if suffix in file_name:
                    f_list.append(file_name)
    if exclude_strings is not None:
        f_list = [f_name for f_name in f_list if all(excluded not in f_name for excluded in exclude_strings)]
    return f_list


def count_files_in_dir(dir_path, suffix):
    """:return number of files with same suffixes in dir (avoiding subdirectories)"""
    if dir_path is None or dir_path is '':
        return None
    only_files = get_files_list(dir_path, suffix)
    return len(only_files)


def my_mkdir(d):
    """only works on *nix system"""
    if not os.path.isdir(d) and not os.path.exists(d):
        os.mkdir(d)


def get_suffix(filename, p='.'):
    """a.jpg -> jpg"""
    pos = filename.rfind(p)
    if pos == -1:
        return ''
    return filename[pos + 1:]


def get_prefix(filename, p='.'):
    """a.jpg -> a"""
    pos = filename.rfind(p)
    if pos == -1:
        return ''
    return filename[:pos]


def model_load(fp):
    suffix = get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        with open(fp, 'rb') as f:
            d = pickle.load(f)
        return d


def model_dump(wfp, obj, append=False):
    suffix = get_suffix(wfp)
    if suffix == 'npy':
        np.save(wfp, obj)
    elif suffix == 'pkl':

        if append:
            style = 'ab'
        else:
            style = 'wb'
        pickle.dump(obj, open(wfp, style))
    else:
        raise Exception('Unknown Type: {}'.format(suffix))
