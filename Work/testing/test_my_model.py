import time

from keras_preprocessing.image import ImageDataGenerator

from consts import DOWNLOAD_FOLDER
from my_models.my_model import MyModel

if __name__ == '__main__':
    model = MyModel(data_path=DOWNLOAD_FOLDER,
                    picture_suffix=['.png', '.jpg'],
                    gpu=True,
                    name='MyFirstTry')

    start = time.time()
    datagen = ImageDataGenerator(
        shear_range=20,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        data_format='channels_last')

    model.load_data()
    print('> reading images completed! (in: %.2f seconds)' % (time.time() - start))

    start = time.time()
    model.compile_model()
    print('> model creation completed! (in: %.2f seconds)' % (time.time() - start))

    start = time.time()
    model.train_model(datagen)
    print('> model training completed! (in: %.2f seconds)' % (time.time() - start))

    start = time.time()
    model.my_save()
    print('> model saving completed! (in: %.2f seconds)' % (time.time() - start))

    start = time.time()
    model.model_predict()
    print('> model prediction completed! (in: %.2f seconds)' % (time.time() - start))
