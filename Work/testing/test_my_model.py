import time

from matplotlib import pyplot as plt

from consts import DOWNLOAD_FOLDER
from image_utils import clean_noise
from my_models import MyModel

if __name__ == '__main__':
    model = MyModel(data_path=DOWNLOAD_FOLDER,
                    picture_suffix=['.png', '.jpg'],
                    gpu=True,
                    path_to_model='../Res101V2_Flatten(linear)',
                    name='Res101V2_Flatten(linear)')

    # name = 'V2Res101_Flatten_ Dropout(5)_10(linear)')

    # test again
    # name='Res101_Flatten_ Dropout(3)_20(relu)_10(linear)')
    start = time.time()
    datagen_args = dict(
        shear_range=25,
        rotation_range=25,
        horizontal_flip=True,
        width_shift_range=5.0,
        height_shift_range=5.0,
        preprocessing_function=lambda x: clean_noise(x, 'float32'),
        # featurewise_center=True,
        # samplewise_center=True,
        # rescale=1.0 / 255,
        dtype='float32',
        data_format='channels_last')
    models = [
        model,
        # model2,
        # model3,
        # model4,
        # model5
    ]
    for m in models:
        print('Model: ' + m.name)
        print('> reading images completed! (in: %.2f seconds)' % (time.time() - start))

        start = time.time()
        model.compile_model()
        print('> model creation completed! (in: %.2f seconds)' % (time.time() - start))

        start = time.time()
        model.my_load("../Res101V2_Flatten(linear).pkl")
        poseGenerator, validation_data = model.train_model(datagen_args)
        print('> model training completed! (in: %.2f seconds)' % (time.time() - start))

        start = time.time()
        model.my_save()
        print('> model saving completed! (in: %.2f seconds)' % (time.time() - start))
        plt.show()

        start = time.time()
        model.eval_model(poseGenerator, validation_data)
        print('> model evaluate completed! (in: %.2f seconds)' % (time.time() - start))

        start = time.time()
        model.model_predict(plot=False)
        print('> model prediction completed! (in: %.2f seconds)' % (time.time() - start))
