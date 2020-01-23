import os

import numpy as np

from consts import DataFileConsts as dConst
from mytools import get_files_list, get_prefix, get_suffix

if __name__ == '__main__':

    folder = dConst.DOWNLOAD_FOLDER
    output_folder = dConst.OUTPUT_FOLDER
    output_path = folder + output_folder

    output_file = dConst.OUTPUT_FILE_NAME

    data_sets_list = dConst.ALL_DATASETS

    suffixes = [data_set['IMAGE_SUFFIX'] for data_set in data_sets_list]
    files_list = get_files_list(output_path, suffixes)

    input_file_list = []

    for f_name in files_list:
        img_prefix_path = get_prefix(f_name)
        img_key = get_suffix(img_prefix_path, '\\')
        lmarks_path = img_prefix_path + '.pts'

        if not os.path.exists(lmarks_path):
            # check if file exits
            lmarks_path = 'None'

        # image_key, path to image, path to landmarks
        input_file_list.append((img_key, f_name, lmarks_path))

    input_file_list = np.asarray(input_file_list, dtype=np.str)
    input_file_list = np.reshape(input_file_list, (len(input_file_list), 3))

    output_file_path = folder + dConst.INPUT_FILE_NAME

    np.savetxt(output_file_path, input_file_list, fmt='%s,%s,%s')
    print('> Wrote %d lines to: %s' % (len(input_file_list), output_file_path))
