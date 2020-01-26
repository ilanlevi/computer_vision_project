import cv2
import numpy as np
from data.gen_model_data import GenKerasModelData

from consts.fpn_model_consts import FPNConsts
from models.fpn_wrapper import load_fpn_model, get_3d_pose
from mytools import get_landmarks2, get_prefix


# todo - delete

def only_once(image_path):
    model_path = FPNConsts.THIS_PATH + FPNConsts.MODELS_DIR
    model_file_name = FPNConsts.POSE_P
    model_name = FPNConsts.MODEL_NAME

    camera_matrix, model_matrix = load_fpn_model(model_path, model_file_name, model_name)

    ldmk_list = get_landmarks2(image_path)
    ldmk_list = np.asarray(ldmk_list)

    rx, ry, rz, tx, ty, tz = get_3d_pose(camera_matrix, model_matrix, ldmk_list)
    this_score = np.asarray([rx, ry, rz, tx, ty, tz])
    out = get_prefix(image_path)
    np.savetxt(out + '.pose', [this_score], fmt='%.4f', delimiter=', ',
               header='pitch, yaw, roll, tx, ty, tz')


if __name__ == '__main__':
    path = 'C:\\Work\\ComputerVision\\datasets\\DATA\\tmp\\'

    out = 'C:\\Work\\ComputerVision\\Project\\tmp\\'

    prefix = '10___image_003_1'

    # only_once(path + prefix + '.d')

    gen = GenKerasModelData(path, dim=500, batch_size=20, to_fit=True, gen_more=True)
    X, Y = gen.__getitem__(0)

    for index in range(len(X)):
        x = X[index]
        y = Y[index]
        to_save = np.asarray([y[0], y[1], y[2], y[3], y[4], y[5]], dtype=np.float)
        np.savetxt(out + str(index) + prefix + '.pose', [to_save], fmt='%.4f', delimiter=', ',
                   header='pitch, yaw, roll, tx, ty, tz')
        x = np.reshape(x, (500, 500))
        cv2.imwrite(out + str(index) + prefix + '.jpg', x)

    print(" >> done")
