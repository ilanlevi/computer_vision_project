import os
import numpy as np
import pickle
import scipy.io as sio


def mkdir(d):
    """only works on *nix system"""
    if not os.path.isdir(d) and not os.path.exists(d):
        os.system('mkdir -p {}'.format(d))


def _get_suffix(filename, p='.'):
    """a.jpg -> jpg"""
    pos = filename.rfind(p)
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        with open(fp, 'rb') as f:
            d = pickle.load(f)
        return d


def _dump(wfp, obj):
    suffix = _get_suffix(wfp)
    if suffix == 'npy':
        np.save(wfp, obj)
    elif suffix == 'pkl':

        pickle.dump(obj, open(wfp, 'wb'))
    else:
        raise Exception('Unknown Type: {}'.format(suffix))


def load_bfm(model_path):
    suffix = _get_suffix(model_path)
    if suffix == 'mat':
        C = sio.loadmat(model_path)
        model = C['model_refine']
        model = model[0, 0]

        model_new = {}
        w_shp = model['w'].astype(np.float32)
        model_new['w_shp_sim'] = w_shp[:, :40]
        w_exp = model['w_exp'].astype(np.float32)
        model_new['w_exp_sim'] = w_exp[:, :10]

        u_shp = model['mu_shape']
        u_exp = model['mu_exp']
        u = (u_shp + u_exp).astype(np.float32)
        model_new['mu'] = u
        model_new['tri'] = model['tri'].astype(np.int32) - 1

        # flatten it, pay attention to index value
        keypoints = model['keypoints'].astype(np.int32) - 1
        keypoints = np.concatenate((3 * keypoints, 3 * keypoints + 1, 3 * keypoints + 2), axis=0)

        model_new['keypoints'] = keypoints.T.flatten()

        #
        w = np.concatenate((w_shp, w_exp), axis=1)
        w_base = w[keypoints]
        w_norm = np.linalg.norm(w, axis=0)
        w_base_norm = np.linalg.norm(w_base, axis=0)

        dim = w_shp.shape[0] // 3
        u_base = u[keypoints].reshape(-1, 1)
        w_shp_base = w_shp[keypoints]
        w_exp_base = w_exp[keypoints]

        model_new['w_norm'] = w_norm
        model_new['w_base_norm'] = w_base_norm
        model_new['dim'] = dim
        model_new['u_base'] = u_base
        model_new['w_shp_base'] = w_shp_base
        model_new['w_exp_base'] = w_exp_base

        _dump(model_path.replace('.mat', '.pkl'), model_new)
        return model_new
    else:
        return _load(model_path)
