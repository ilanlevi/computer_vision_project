class MyModelConsts:
    # const my model model
    def __init__(self):
        pass

    MODEL_DIR = "C:\\Work\\ComputerVision\\Project\\blabla"

    # stats file consts
    MODEL_STATES_FILE = "states.npy"
    MODEL_STATES_FILE_PATH = MODEL_DIR + '/' + MODEL_STATES_FILE

    # model file consts
    # todo - delete
    MODEL_NAME_80_4l = ("myModel_80_4l.h5", 80, 4)
    MODEL_NAME_160_4l = ("myModel_160_4l.h5", 160, 4)
    MODEL_NAME_80_3l = ("myModel_80_3l.h5", 80, 3)
    MODEL_NAME_160_3l = ("myModel_160_4l.h5", 160, 3)

    # MODEL_NAME = ("myModel_160_4l.h5", 160, 3)

    ALL = [MODEL_NAME_80_3l, MODEL_NAME_160_3l, MODEL_NAME_80_4l, MODEL_NAME_160_4l]

    # OPTIONS = [M]

    MODEL_NAME_PATH = MODEL_DIR + '/' + 'Model.h5'

    BATCH_SIZE = 64
    EPOCHS = 120
