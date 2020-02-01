import time

from consts import DOWNLOAD_FOLDER, OUTPUT_FOLDER
from my_models.my_model import MyModel

if __name__ == '__main__':
    model = MyModel(data_path=DOWNLOAD_FOLDER + OUTPUT_FOLDER,
                    picture_suffix=['.png', '.jpg'],
                    gpu=True,
                    name='MyFirstTry')

    start = time.time()
    model.load_data()

    print('> reading images completed! (in: %.2f seconds)' % (time.time() - start))
    start = time.time()

    model.compile_model()

    print('> model creation completed! (in: %.2f seconds)' % (time.time() - start))
    start = time.time()

    model.train_model()

    print('> model training completed! (in: %.2f seconds)' % (time.time() - start))
    start = time.time()

    model.model_predict()

    print('> model prediction completed! (in: %.2f seconds)' % (time.time() - start))
