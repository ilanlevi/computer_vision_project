import time

import urllib


def download_data(download_urls, file_names, to_path='', print_data=False):
    """
    Downloads data from given urls to given files name using urllib
    :param download_urls: the url list or single url
    :param file_names: the download files names on local pc list or single url (same length as download_urls)
    :param to_path: the prefix path to download (can be empty)
    :param print_data: print data or not, default is False
    :return: True if succeed or false other wise
    """

    if print_data:
        print('Starting to download data!')
        print('From: ' + download_urls)
        print('To: ' + file_names)

    # convert to lists if needed
    if not isinstance(download_urls, list):
        download_urls = [download_urls]
    if not isinstance(file_names, list):
        file_names = [file_names]

    # check length
    if len(download_urls) != len(file_names):
        if print_data:
            print('The list sizes doesnt match, ignoring!')
        return False

    start = time.time()

    for download_url, f_name in download_urls, file_names:
        try:
            urllib.urlretrieve(download_url, to_path + f_name)
        except Exception as e:
            if print_data:
                print('Download failed! (in: %.2f seconds)' % (time.time() - start))
                print('Reason: ' + str(e))
            return False

    if print_data:
        print('Download completed! (in: %.2f seconds)' % (time.time() - start))
    return True
