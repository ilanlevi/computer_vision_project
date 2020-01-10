class DataFileConsts:
    def __init__(self):
        pass

    DOWNLOAD_FOLDER = 'C:\\Work\\ComputerVision\\datasets\\'

    CONVERTED_FILE_NAME = 'formatted.csv'

    # helen dataset consts
    HELEN_FOLDER = 'helen\\'

    NUMBER_OF_HELEN_LINKS = 5

    HELEN_INFO_INFO = 'http://www.ifp.illinois.edu/~vuongle2/helen/'

    HELEN_FILES = ['helen_%d.zip' % i for i in range(1, 6)]

    HELEN_DOWNLOAD_URLS = [
        'http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_' + filename for filename in HELEN_FILES
    ]

    HELEN_FORMATTED = 'helen_' + CONVERTED_FILE_NAME




