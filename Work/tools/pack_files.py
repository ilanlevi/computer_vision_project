import gzip
import shutil
import time


class DataZip:

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
            with gzip.open(self.from_file, 'r') as zip_ref:
                data = zip_ref.read()
                with open(self.to_dir, 'wb') as f_out:
                    f_out.write(data)
            # both files are closed at this point
        except Exception as e:
            print 'Extracting failed! (in: %.2f seconds)' % (time.time() - start)
            print 'Reason: ' + str(e)
            return False

        print 'Extracting completed! (in: %.2f seconds)' % (time.time() - start)
        return True

    # def
