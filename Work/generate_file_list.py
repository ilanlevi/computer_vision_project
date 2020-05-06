import argparse

from consts import PICTURE_SUFFIX
from my_utils import get_files_list

# generate file list from path
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--directory", required=True,
                    help="full path directory - will ignore none ['jpg', 'png'] files")
    args = vars(ap.parse_args())

    files_file_path = args["directory"]
    file_list = get_files_list(files_file_path, PICTURE_SUFFIX)

    out_file_name = 'test_set_input.txt'
    with open(out_file_name, 'w') as out_f:
        for file_name in file_list:
            out_f.write('%s\n' % file_name)
