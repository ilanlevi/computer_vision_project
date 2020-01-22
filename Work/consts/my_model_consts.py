class MyModelConsts:
    # const my model model
    def __init__(self):
        pass

    MODEL_DIR = "/my_model"

    # stats file consts
    MODEL_STATES_FILE = "states.npy"
    MODEL_STATES_FILE_PATH = MODEL_DIR + '/' + MODEL_STATES_FILE

    # model file consts
    MODEL_NAME = "myModel.h5"
    MODEL_NAME_PATH = MODEL_DIR + '/' + MODEL_NAME

    BATCH_SIZE = 64
    EPOCHS = 100