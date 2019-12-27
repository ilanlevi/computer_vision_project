import fnmatch
import os

import numpy as np

from Work.consts.files_consts import FileConsts
from Work.tools.download_data import DownloadData
from Work.tools.pack_files import DataZip


class HandleDownloadData:

    def __init__(self, download_urls=FileConsts.DS_DOWNLOAD_URLS,
                 to_dir=FileConsts.DOWNLOADED_DIR):
        self.download_urls = np.asarray(download_urls)
        self.to_dir = to_dir

    def download_all(self):
        action_succeeded = True

        for index in range(len(self.download_urls)):
            download_url = self.download_urls[index][0]
            local_url = self.to_dir + self.download_urls[index][1]
            data = DownloadData(download_url, local_url)
            action_succeeded = action_succeeded & data.download_data()

        return action_succeeded

    def unzip_files(self):
        action_succeeded = True

        files = os.listdir(self.to_dir)
        for file_name in files:
            if 'zip' in file_name:
                unzip = DataZip(self.to_dir + file_name, self.to_dir)
                action_succeeded = action_succeeded & unzip.unzipped_data()

        return action_succeeded

    def download_unpack(self, download_files=True, unzip_file=True):
        """
             Downloads data and unzip files if needed
             :return: True if succeed or false other wise
        """

        if download_files:
            action_succeeded = self.download_all()
        else:
            action_succeeded = True

        if not action_succeeded:
            return action_succeeded

        if unzip_file:
            action_succeeded = self.unzip_files()

        if not action_succeeded:
            return action_succeeded

        return action_succeeded


if __name__ == '__main__':
    """
    Download and unpack default files 
    """
    success = HandleDownloadData().download_unpack(download_files=False)
    print 'Action succeeded = ' + str(success)
