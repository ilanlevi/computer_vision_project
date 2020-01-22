from models.my_model import MyModel
from consts import DataFileConsts as dsC
import time

if __name__ == '__main__':
    model = MyModel(dsC.DOWNLOAD_FOLDER + dsC.OUTPUT_FOLDER, ['.png', '.jpg'], gpu=True)

    start = time.time()
    model.load_data()
    model.pre_process_data()

    print('> reading images completed! (in: %.2f seconds)' % (time.time() - start))
    start = time.time()

    model.create()

    print('> model creation completed! (in: %.2f seconds)' % (time.time() - start))
    start = time.time()

    model.train_model()

    print('> model training completed! (in: %.2f seconds)' % (time.time() - start))
    start = time.time()

    model.model_predict()

    print('> model prediction completed! (in: %.2f seconds)' % (time.time() - start))
    start = time.time()
