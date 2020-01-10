class DataFileConsts:
    def __init__(self):
        pass

    DOWNLOAD_FOLDER = 'C:\\Work\\ComputerVision\\datasets\\data\\'

    OUTPUT_FILE_NAME = 'scores.csv'

    # helen dataset consts
    NUMBER_OF_HELEN_LINKS = 5

    HELEN_INFO_URL = 'http://www.ifp.illinois.edu/~vuongle2/helen/'

    HELEN_DATA = {
        'FOLDER': 'helen\\',
        'ANNOTATIONS': 'annotations\\',
        'FILES': ['helen_%d.zip' % i for i in range(1, 1 + NUMBER_OF_HELEN_LINKS)],
        'DOWNLOAD_URLS': [
            HELEN_INFO_URL + 'data/helen_' + 'helen_%d.zip' % i for i in range(1, 1 + NUMBER_OF_HELEN_LINKS)
        ],
        'ANNOTATIONS_DOWNLOAD': (HELEN_INFO_URL + 'data/annotation.zip'),
        'FORMATTED': 'helen_' + CONVERTED_FILE_NAME,
        'IMAGE_SUFFIX': '.jpg'
    }

