import time

from consts import DataFileConsts as dsC
from models.my_model import MyModel

if __name__ == '__main__':
    model = MyModel(data_path=dsC.DOWNLOAD_FOLDER + dsC.OUTPUT_FOLDER, picture_suffix=['.png', '.jpg'], gpu=True,
                    image_size=160, name='2Conv32_Conv64_flatten_2dense')

    start = time.time()
    model.load_data()

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
