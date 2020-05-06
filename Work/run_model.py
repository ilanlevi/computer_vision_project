import argparse
import os
import time

import numpy as np

from consts import NEW_MY_MODEL_LOCAL_PATH
from image_utils import clean_noise, FpnWrapper
from my_models import MyModel

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="full path to input.txt list file - each image in a new line")
    ap.add_argument("-o", "--output", required=True,
                    help="full path to output file (csv)")
    args = vars(ap.parse_args())

    files_file_path = args["dataset"]

    with open(files_file_path) as f:
        rows = [rows.strip() for rows in f]

    rows = np.asarray(rows)
    all_files = []
    for row in rows:
        if '#' not in row:
            all_files.append(row)

    if all_files is None or len(all_files) is 0:
        raise IOError('Input file is empty!')

    print('Found total of: %d files!' % len(rows))

    start = time.time()
    print('>> Loading model!')

    model = MyModel(os.getcwd(),
                    all_files,
                    path_to_model=NEW_MY_MODEL_LOCAL_PATH,
                    name="Res101V2_Flatten(linear)",
                    gpu=False,
                    )

    fpn_wrapper = FpnWrapper(path_to_model='')
    model.my_load("Res101V2_Flatten(linear).pkl")
    print('>> Loaded in: %.2f seconds! ' % (time.time() - start))
    start = time.time()
    settings = dict(
        preprocessing_function=lambda x: clean_noise(x, 'float32'),
        dtype='float32',
        data_format='channels_last')

    model.model_predict(all_files, args["output"], datagen_args=settings, fpn_wrapper=fpn_wrapper)
    print('>> model prediction completed! (in: %.2f seconds)' % (time.time() - start))

    # # print score diff - this is just for testing
    # my_score_file = args["output"]
    # my_diff_file = 'diff.csv'
    # compare_scores_no_translation('', 'valid_set2.csv', my_score_file, my_diff_file, True)
    # plot_diff_no_translation('', my_diff_file, title='diff')
    # plot_diff_each_param_no_translation('', ['valid_set2.csv', my_score_file])
    #
    # plt.show()
