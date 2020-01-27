# todo - merger with ds_consts
class DataFileConsts:
    def __init__(self):
        pass

    DOWNLOAD_FOLDER = 'C:\\Work\\ComputerVision\\datasets\\data\\'

    OUTPUT_FOLDER = 'output\\'

    OUTPUT_FILE_SUFFIX = '.param'
    OUTPUT_FILE_NAME = 'scores.csv'

    INPUT_FILE_NAME = 'input.txt'

    # all of the data was downloaded from:
    # https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

    W300_DATA = {
        'FOLDER': '300W',
        'IMAGE_SUFFIX': '.png'
    }

    AFW_DATA = {
        'FOLDER': 'afw',
        'IMAGE_SUFFIX': '.jpg'
    }

    IBUG_DATA = {
        'FOLDER': 'ibug',
        'IMAGE_SUFFIX': '.jpg',
    }

    HELEN_DATA = {
        'FOLDER': 'helen',
        'IMAGE_SUFFIX': '.jpg',
    }

    LFPW_DATA = {
        'FOLDER': 'lfpw',
        'IMAGE_SUFFIX': '.png',
    }

    ALL_DATASETS = [W300_DATA, AFW_DATA, IBUG_DATA, HELEN_DATA, LFPW_DATA]
