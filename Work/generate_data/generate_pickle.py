import time

import numpy as np

from consts import DataFileConsts as dsC
from data import ModelData
from mytools.my_io import mkdir


# todo - delete

def update_pickle():
    data = np.load(DATA)
    print(data['arr_0'].mean())


def genenerat_pickle():
    mkdir(d)

    start = time.time()
    dataset = ModelData(dsC.DOWNLOAD_FOLDER + dsC.OUTPUT_FOLDER, batch_size=1, image_size=160,
                        picture_suffix=['.png', '.jpg'],
                        train_rate=1, to_gray=True, to_hog=True, sigma=0.33).init()

    has_more = dataset.read_data_set()
    total_size = len(dataset.original_file_list)
    while has_more:
        print("We are in: %d (total = %d), left = %.2f %s" % (
            dataset.read_index, total_size, dataset.read_index * 100 / total_size, '%'))
        # dataset.normalize_data()
        # dataset.canny_filter()
        with open(DATA, 'ab') as f:
            np.savez_compressed(f, x=dataset.x_train_set, y=dataset.y_train_set)
        has_more = dataset.read_data_set()

    print('> reading images completed! (in: %.2f seconds)' % (time.time() - start))


d = 'C:/Work/ComputerVision/Project/Work/pkldata/'
DATA = d + 'DATA.npz'
LABEL = d + 'Y.npz'
GENERATE_PICKLE = False
UPDATE_PICKLE = True

if __name__ == '__main__':
    if GENERATE_PICKLE:
        genenerat_pickle()
    if UPDATE_PICKLE:
        update_pickle()
