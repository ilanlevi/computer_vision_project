import os
import pickle

import numpy as np


def get_files_list(path, suffixes=None):
    """
        return file list from path with the same
         suffixes (if none all of the returned)
    """

    if suffixes is None:
        suffixes = []

    if not isinstance(suffixes, list):
        suffixes = [suffixes]

    files = [os.path.join(r, file_) for r, d, f in os.walk(path) for file_ in f]
    f_list = []
    for file_name in files:
        if len(suffixes) == 0:
            f_list.append(file_name)
        else:
            for suffix in suffixes:
                if suffix in file_name:
                    f_list.append(file_name)
    return f_list


def mkdir(d):
    """only works on *nix system"""
    if not os.path.isdir(d) and not os.path.exists(d):
        os.system('mkdir -p {}'.format(d))


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


def model_dump(wfp, obj):
    suffix = get_suffix(wfp)
    if suffix == 'npy':
        np.save(wfp, obj)
    elif suffix == 'pkl':
        pickle.dump(obj, open(wfp, 'wb'))
    else:
        raise Exception('Unknown Type: {}'.format(suffix))
