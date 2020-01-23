import time

import cv2
from matplotlib import pyplot as plt

from image_utils.image_tools import auto_canny

# todo - delete

if __name__ == '__main__':
    start_time = time.time()

    im = cv2.imread(
        "C:/Work/ComputerVision/datasets/DATA/output/74___2231468853_1/74___2231468853_1_rendered_aug_-00_00_01.jpg", 1)
    # cv2.imshow('', im)
    v = auto_canny(im, 0.33)
    plt.imshow(v)
    v2 = auto_canny(im, 0.001)
    plt.figure()
    plt.imshow(v2)
    plt.show()
    print('blabla')
