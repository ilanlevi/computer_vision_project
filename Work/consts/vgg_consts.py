class HelenFileConsts:
    def __init__(self):
        pass

    DS_DOWNLOAD_URLS = [
        ('https://www.robots.ox.ac.uk/~vgg/data/vgg_face/vgg_face_dataset.tar.gz', 'vgg_face_dataset.tar.gz'),
    ]

    FACES_DATASET_INFO = 'https://www.robots.ox.ac.uk/~vgg/data/vgg_face/'

    DOWNLOAD_FOLDER = 'C:\\Work\\ComputerVision\\'
    DOWNLOAD_SUB_FOLDER = 'vgg\\'

    PROCESSED_SET_FOLDER = 'processed_vgg\\'

    DOWNLOADED_DIR = DOWNLOAD_FOLDER + DOWNLOAD_SUB_FOLDER

    PREDICTOR_FILE_NAME = 'shape_predictor_68_face_landmarks.dat'
