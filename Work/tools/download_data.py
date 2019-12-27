import time

import urllib


class DownloadData:

    def __init__(self, download_url, to_file):
        self.download_url = download_url
        self.to_file = to_file

    def download_data(self):
        """
        Downloads data from given url to given file name using urllib
        :return: True if succeed or false other wise
        """

        print 'Starting to download data!'
        print 'From: ' + self.download_url
        print 'To: ' + self.to_file

        start = time.time()

        try:
            urllib.urlretrieve(self.download_url, self.to_file)
        except Exception as e:
            print 'Download failed! (in: %.2f seconds)' % (time.time() - start)
            print 'Reason: ' + str(e)
            return False

        print 'Download completed! (in: %.2f seconds)' % (time.time() - start)
        return True
