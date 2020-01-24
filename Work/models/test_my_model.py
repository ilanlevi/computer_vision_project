import time

from consts import DataFileConsts as dsC
from consts import MyModelConsts as modelC
from models.my_model import MyModel

if __name__ == '__main__':

    for model_name, model_size, densed_num in modelC.ALL:
        print(">>> Starting model: " + model_name)
        model = MyModel(data_path=dsC.DOWNLOAD_FOLDER + dsC.OUTPUT_FOLDER, picture_suffix=['.png', '.jpg'], gpu=True
                        , number_of_dense=densed_num, image_size=model_size, name=model_name)

        start = time.time()
        model.load_data(image_size=model_size)

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
