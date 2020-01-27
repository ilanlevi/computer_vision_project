class DataSetConsts:
    # const
    DEFAULT_RANDOM_STATE = 0
    DEFAULT_TRAIN_RATE = 0.7

    DEFAULT_TEST_RATE = 0.3
    DEFAULT_VALID_RATE = 0.01

    PICTURE_WIDTH = 500
    # PICTURE_SIZE = 160
    LANDMARKS_SHAPE = (68, 2)
    PICTURE_SIZE = 100
    BATCH_SIZE = 64
    OUT_DIM = 6

    SIGMA = 0.33

    LANDMARKS_PREFIX = '_landmarks_'

    LANDMARKS_FILE_SUFFIX = '.pts'

    PICTURE_SUFFIX = ['jpg', 'png']

    def __init__(self):
        pass
