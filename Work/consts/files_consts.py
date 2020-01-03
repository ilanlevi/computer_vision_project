class FileConsts:
    def __init__(self):
        pass

    DS_DOWNLOAD_URLS = [
        ('http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_1.zip', 'helen_1.zip'),
        ('http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_2.zip', 'helen_2.zip'),
        ('http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_3.zip', 'helen_3.zip'),
        ('http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_4.zip', 'helen_4.zip'),
        ('http://www.ifp.illinois.edu/~vuongle2/helen/data/helen_5.zip', 'helen_5.zip'),
    ]

    FACES_DATASET_INFO = 'http://www.ifp.illinois.edu/~vuongle2/helen/'

    DOWNLOAD_FOLDER = 'C:\\Work\\ComputerVision\\'

    DOWNLOAD_SUB_FOLDER = 'db\\'

    VALID_SET_SUB_FOLDER = 'images\\'

    PROCESSED_SET_FOLDER = 'processed\\'

    DOWNLOADED_DIR = DOWNLOAD_FOLDER + DOWNLOAD_SUB_FOLDER

    PREDICTOR_FILE_NAME = 'shape_predictor_68_face_landmarks.dat'

    SETTING_PKL_FILE_NAME = 'param_whitening1.pkl'

    VALIDATION_FOLDER = 'C:\\Work\\ComputerVision\\valid_set\\valid_set\\'

    VALIDATION_CSV = 'validation_set.csv'
    MY_VALIDATION_CSV = 'validation_set_my.csv'
    VALIDATION_DIFF_CSV = 'validation_set_diff.csv'

    MY_VALIDATION_CSV2 = 'validation_set_my2.csv'
    VALIDATION_DIFF_CSV2 = 'validation_set_diff2.csv'
    VALIDATION_DIFF_CSV3 = 'validation_set_diff3.csv'
