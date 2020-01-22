from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.models import load_model, save_model, Sequential
from sympy.printing.theanocode import theano

from consts import MyModelConsts as myC, DataSetConsts as dsC
from data import ModelData
from matplotlib import pyplot as plt
from mytools import write_csv, mkdir


class MyModel:

    def __init__(self, data_path, picture_suffix, rate=dsC.DEFAULT_TRAIN_RATE, image_size=dsC.PICTURE_SIZE,
                 sigma=0.33, path=myC.MODEL_DIR, name=myC.MODEL_NAME, gpu=False, batch_size=myC.BATCH_SIZE,
                 epochs=myC.EPOCHS):
        """
        :type self.data: ModelData
        """
        self.data = None
        self.data_path = data_path
        self.picture_suffix = picture_suffix

        if not isinstance(picture_suffix, list):
            self.picture_suffix = [picture_suffix]

        self.rate = rate
        self.image_size = image_size
        self.sigma = sigma

        self.path = path
        self.name = name
        self.model = None
        self.gpu = gpu

        self.epochs = epochs
        self.batch_size = batch_size

        # set keras config
        theano.config.floatX = 'float32'
        if self.gpu:
            theano.config.device = 'gpu'

    def create(self):
        self.model = Sequential()
        size = (self.data.x_train_set.shape[0], self.data.x_train_set.shape[1])
        self.model.add(Dense(units=150, activation='relu', kernel_regularizer='l2', input_dim=size))
        self.model.add(Dense(units=40, activation='relu', kernel_regularizer='l2'))
        self.model.add(Dense(units=20, activation='relu', kernel_regularizer='l2'))
        self.model.add(Dense(units=6, activation='linear'))

        print(self.model.summary())

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def load(self):
        self.model = load_model(self.get_full_path())

    def save(self):
        mkdir(self.path)
        save_model(self.get_full_path())

    def get_full_path(self):
        return self.path + '/' + self.name

    def load_data(self, data_path=None, image_size=None, picture_suffix=None, rate=None, sigma=None):
        if data_path is None:
            data_path = self.data_path
        if image_size is None:
            image_size = self.image_size
        if picture_suffix is None:
            picture_suffix = self.picture_suffix
        if rate is None:
            rate = self.rate
        if sigma is None:
            sigma = self.sigma

        self.data = ModelData(data_path=data_path, image_size=image_size, picture_suffix=picture_suffix,
                              train_rate=rate,
                              to_gray=True, to_hog=True, sigma=sigma)

    def pre_process_data(self):
        self.data.init()
        self.data.split_dataset()
        self.data.normalize_data()
        self.data.canny_filter()

    def train_model(self, save=True, plot=True):

        callback_list = [EarlyStopping(monitor='val_loss', patience=25)]

        hist = self.model.fit(x=self.data.x_train_set, y=self.data.y_train_set,
                              validation_data=(self.data.x_valid_set, self.data.y_valid_set),
                              batch_size=self.batch_size, epochs=self.epochs,
                              callbacks=callback_list)

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
        y_pred = self.model.predict(self.data.x_test_set)
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
