import time
import zipfile


def unzipped_data(files, to_dir=None, print_data=False):
    """
    unzip data from given url to given file name that is in self.dir
    :param files: the files list or single url
    :param to_dir: the directory to extract the data, default is None
    :param print_data: print data or not, default is False
    :return: True if succeed or false other wise
    """
    if print_data:
        print 'Starting to unzip data!'
        print 'From: ' + files
        print 'To: ' + to_dir

    # convert to lists if needed
    if not isinstance(files, list):
        files = [files]

    start = time.time()

    for from_file in files:
        try:
            with zipfile.ZipFile(from_file, 'r') as zip_ref:
                zip_ref.extractall(to_dir)
                zip_ref.close()
            # both files are closed at this point
        except Exception as e:
            if print_data:
                print 'Extracting failed! (in: %.2f seconds)' % (time.time() - start)
                print 'Reason: ' + str(e)
            return False

    if print_data:
        print 'Extracting completed! (in: %.2f seconds)' % (time.time() - start)

    return True
