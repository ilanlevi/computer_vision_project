import time
from multiprocessing import Pool
from random import randint

import numpy as np

from consts.datasets_consts import DataFileConsts as dConsts
from consts.fpn_model_consts import FPNConsts
from data.labeled_data import LabeledData
from models.fpn_wrapper import load_fpn_model, get_3d_pose
from utils.image_tools import save_images, auto_canny
from  matplotlib import pyplot as plt
import cv2

# todo - delete

if __name__ == '__main__':
    start_time = time.time()

    im = cv2.imread("C:/Work/ComputerVision/datasets/DATA/output/74___2231468853_1/74___2231468853_1_rendered_aug_-00_00_01.jpg", 1)
    # cv2.imshow('', im)
    v = auto_canny(im, 0.33)
    plt.imshow(v)
    v2 = auto_canny(im, 0.001)
    plt.figure()
    plt.imshow(v2)
    plt.show()
    print 'blabla'
