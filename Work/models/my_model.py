import os

from keras.callbacks import EarlyStopping
from keras.layers import Dense, MaxPooling2D, Dropout, Conv2D
from keras.models import load_model, save_model, Sequential
from matplotlib import pyplot as plt

from consts import MyModelConsts as myC, DataSetConsts as dsC
from data import KerasModelData
from mytools import mkdir


class MyModel:

    def __init__(self, data_path, picture_suffix, test_rate=dsC.DEFAULT_TRAIN_RATE, valid_rate=dsC.DEFAULT_VALID_RATE,
                 image_size=dsC.PICTURE_SIZE, sigma=0.33, path=myC.MODEL_DIR, name=myC.MODEL_NAME, gpu=False,
                 batch_size=myC.BATCH_SIZE, epochs=myC.EPOCHS):
        self.data = None
        self.test_data = None
        self.labels = None
        self.data_path = data_path
        self.picture_suffix = picture_suffix

        if not isinstance(picture_suffix, list):
            self.picture_suffix = [picture_suffix]

        self.test_rate = test_rate
        self.valid_rate = valid_rate
        self.image_size = image_size
        self.sigma = sigma

        self.path = path
        self.name = name
        self.model = None
        self.gpu = gpu

        self.epochs = epochs
        self.batch_size = batch_size

        # set keras config
        # theano.config.floatX = 'float32'
        # todo
        if self.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    def create(self):
        self.model = Sequential()

        # self.model.add(Dense(units=120, activation='relu', kernel_regularizer='l2', input_dim=160 * 160))
        conv_size = int(self.image_size * 0.8)
        self.model.add(Conv2D(conv_size, (3, 3), activation='relu', input_shape=(1, self.image_size, self.image_size)))
        self.model.add(Conv2D(conv_size, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(units=120, activation='relu', kernel_regularizer='l2'))
        self.model.add(Dense(units=40, activation='relu', kernel_regularizer='l2'))
        self.model.add(Dense(units=20, activation='relu', kernel_regularizer='l2'))
        self.model.add(Dense(units=6, activation='linear'))

        print(self.model.summary())

        self.model.compile(optimizer='adam', loss='mean_squared_error', use_multiprocessing=True)

    def load(self):
        self.model = load_model(self.get_full_path())

    def save(self):
        mkdir(self.path)
        save_model(self.model, self.get_full_path())

    def get_full_path(self):
        return self.path + '/' + self.name

    def load_data(self, data_path=None, image_size=None, picture_suffix=None, t_rate=None, v_rate=None, sigma=None):
        if data_path is None:
            data_path = self.data_path
        if image_size is None:
            image_size = self.image_size
        if picture_suffix is None:
            picture_suffix = self.picture_suffix
        if t_rate is None:
            t_rate = self.test_rate
        if v_rate is None:
            v_rate = self.valid_rate
        if sigma is None:
            sigma = self.sigma

        self.data = KerasModelData(data_path=data_path, dim=image_size, picture_suffix=picture_suffix,
                                   to_gray=True, to_hog=True, sigma=sigma, test_rate=t_rate, valid_rate=v_rate)

    def train_model(self, save=True, plot=True):

        test_files, validation_files = self.data.split_to_train_and_validation()

        self.test_data = KerasModelData(self.data_path, original_file_list=test_files)
        validation_data = KerasModelData(self.data_path, original_file_list=validation_files)

        callback_list = [EarlyStopping(monitor='val_loss', patience=25)]

        hist = self.model.fit_generator(generator=self.data, validation_data=validation_data, callbacks=callback_list)

        if save:
            self.save()

        print()
        print('Train loss:', self.model.evaluate(self.data.x_train_set, self.data.y_train_set, verbose=0))
        print('  Val loss:', self.model.evaluate(self.data.x_valid_set, self.data.y_valid_set, verbose=0))
        print(' Test loss:', self.model.evaluate(self.data.x_test_set, self.data.y_test_set, verbose=0))

        if plot:
            history = hist.history
            loss_train = history['loss']
            loss_val = history['val_loss']

            plt.figure()
            plt.plot(loss_train, label='train')
            plt.plot(loss_val, label='val_loss', color='red')
            plt.legend()

    def model_predict(self, plot=True, save_score=True):
        if self.test_data is None:
            self.test_data = self.data

        y_pred = self.model.predict_generator(self.test_data)
        diff = self.data.y_test_set - y_pred

        if plot:
            diff_roll = diff[:, 0]
            diff_pitch = diff[:, 1]
            diff_yaw = diff[:, 2]

            plt.figure(figsize=(16, 10))

            plt.subplot(3, 1, 1)
            plt.plot(diff_roll, color='red')
            plt.title('roll')

            plt.subplot(3, 1, 2)
            plt.plot(diff_pitch, color='red')
            plt.title('pitch')

            plt.subplot(3, 1, 3)
            plt.plot(diff_yaw, color='red')
            plt.title('yaw')

            plt.tight_layout()
        #
        # if save_score:
        #     mkdir(self.path)
        #     csv_data = []
        #     for index in range(len(y_pred)):
        #         csv_data.append()
        #     write_csv()
