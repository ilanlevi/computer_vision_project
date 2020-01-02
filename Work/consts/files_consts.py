class HelenFileConsts:
    DS_DOWNLOAD_URLS = [
        ('http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_1.zip', 'helen_1.zip'),
        ('http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_2.zip', 'helen_2.zip'),
        ('http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_3.zip', 'helen_3.zip'),
        ('http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_4.zip', 'helen_4.zip'),
        ('http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_5.zip', 'helen_5.zip'),
    ]

    FACES_DATASET_INFO = 'http://www.ifp.illinois.edu/~vuongle2/helen/'

    DOWNLOAD_FOLDER = 'C:\\Work\\ComputerVision\\'
    DOWNLOAD_FOLDER2 = 'C:\\Work\\ComputerVision\\valid_set\\valid_set\\'
    DOWNLOAD_SUB_FOLDER = 'db\\'
    VALID_SET_SUB_FOLDER = 'images\\'

    PROCESSED_SET_FOLDER = 'processed\\'

    DOWNLOADED_DIR = DOWNLOAD_FOLDER + DOWNLOAD_SUB_FOLDER

    PREDICTOR_FILE_NAME = 'shape_predictor_68_face_landmarks.dat'

    SETTING_PKL_FILE_NAME = 'param_whitening1.pkl'

    def __init__(self, download_folder=DOWNLOAD_FOLDER, downloaded_sub_folder=DOWNLOAD_SUB_FOLDER):
        HelenFileConsts.DOWNLOAD_FOLDER = download_folder
        HelenFileConsts.DOWNLOAD_SUB_FOLDER = downloaded_sub_folder
        HelenFileConsts.DOWNLOADED_DIR = download_folder + downloaded_sub_folder
