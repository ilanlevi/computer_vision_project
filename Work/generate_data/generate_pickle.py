import os
import time

from consts import DataFileConsts as dsC
from data import ModelData
from mytools.my_io import model_dump, mkdir

if __name__ == '__main__':
    d = os.getcwd() + '/dataset/'
    mkdir(d)
    DATA = d + 'X.pkl'
    LABEL = d + 'Y.pkl'

    start = time.time()
    dataset = ModelData(dsC.DOWNLOAD_FOLDER + dsC.OUTPUT_FOLDER, image_size=160, picture_suffix=['.png', '.jpg'],
                        train_rate=1, to_gray=True, to_hog=True, sigma=0.33).init()

    has_more = dataset.read_data_set()
    total_size = len(dataset.original_file_list)
    while has_more:
        print('We are in: %d (total = %d), left = %d' %
              (dataset.read_index, total_size, total_size - dataset.read_index))
        dataset.normalize_data()
        dataset.canny_filter()
        model_dump(DATA, dataset.x_train_set, append=True)
        model_dump(LABEL, dataset.y_train_label, append=True)
        has_more = dataset.read_data_set()

    print('> reading images completed! (in: %.2f seconds)' % (time.time() - start))
