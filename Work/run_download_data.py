from tools.download_data import DownloadData
from tools.pack_files import DataZip
from consts.files_consts import FileConsts
import os


def download_unpack(download_file=True, unzip_file=True, unpack_files=True):
    to_url = os.getcwd() + "\\" + FileConsts.DOWNLOADED_UNTOUCHED_FILE
    if download_file:
        data = DownloadData(FileConsts.FACES_DATASET_DOWNLOAD_URL, to_url)
        download_success = data.download_data()
    else:
        download_success = True
    if unzip_file:
        if download_success:
            z = DataZip(from_file=to_url, to_dir=(os.getcwd() + "\\" + FileConsts.DOWNLOADED_UNTOUCHED_DIR + "\\"))
            return z.unzipped_data()
    return False


if __name__ == '__main__':
    success = download_unpack(False)
    print 'Process succeeded: %s' % str(success)
