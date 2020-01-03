import time
import zipfile


class DataUnzip:

    def __init__(self, from_file, to_dir=''):
        self.from_file = from_file
        self.to_dir = to_dir

    def unzipped_data(self):
        """
        unzip data from given url to given file name that is in self.dir
        :return: True if succeed or false other wise
        """
        print 'Starting to unzip data!'
        print 'From: ' + self.from_file
        print 'To: ' + self.to_dir

        start = time.time()

        try:
            with zipfile.ZipFile(self.from_file, 'r') as zip_ref:
                zip_ref.extractall(self.to_dir)
                zip_ref.close()
            # both files are closed at this point
        except Exception as e:
            print 'Extracting failed! (in: %.2f seconds)' % (time.time() - start)
            print 'Reason: ' + str(e)
            return False

        print 'Extracting completed! (in: %.2f seconds)' % (time.time() - start)
        return True
