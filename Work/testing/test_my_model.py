import time

from consts import DOWNLOAD_FOLDER
from my_models import MyNewModel

if __name__ == '__main__':
    model = MyNewModel(data_path=DOWNLOAD_FOLDER,
                       picture_suffix=['.png', '.jpg'],
                       gpu=True,
                       name='MyFirstTry')

    start = time.time()
    datagen_args = dict(
        shear_range=15,
        rotation_range=40,
        width_shift_range=10,
        height_shift_range=10,
        horizontal_flip=True,
        dtype='float32',
        data_format='channels_last')

    print('> reading images completed! (in: %.2f seconds)' % (time.time() - start))

    start = time.time()
    model.compile_model()
    print('> model creation completed! (in: %.2f seconds)' % (time.time() - start))

    start = time.time()
    model.train_model(datagen_args)
    print('> model training completed! (in: %.2f seconds)' % (time.time() - start))

    start = time.time()
    model.my_save()
    print('> model saving completed! (in: %.2f seconds)' % (time.time() - start))

    start = time.time()
    model.model_predict()
    print('> model prediction completed! (in: %.2f seconds)' % (time.time() - start))
