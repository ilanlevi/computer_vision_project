import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from keras.models import load_model, save_model, Sequential
from matplotlib import pyplot as plt

from consts import DEFAULT_VALID_RATE, MY_MODEL_LOCAL_PATH, MY_MODEL_NAME, BATCH_SIZE, EPOCHS, \
    PICTURE_SUFFIX, PICTURE_SIZE, DEFAULT_TEST_RATE
from my_models import MyDataIterator
from my_utils import mkdir, model_dump, count_files_in_dir


class MyModel:

    def __init__(self,
                 data_path,
                 original_file_list=None,
                 image_size=PICTURE_SIZE,
                 picture_suffix=PICTURE_SUFFIX,
                 test_rate=DEFAULT_TEST_RATE,
                 valid_rate=DEFAULT_VALID_RATE,
                 path_to_model=MY_MODEL_LOCAL_PATH,
                 name=MY_MODEL_NAME,
                 gpu=False,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS):

        """
        Create my model instance
        :param data_path: the data path
        :param original_file_list: if specified this will be the files list and not data_path
        :param image_size: the image size after loading will be (image_size, image_size)
        :param picture_suffix: the images type
        :param test_rate: should be [0.0 ,1.0]. proportion of the dataset test split (only on training)
        :param valid_rate: should be [0.0 ,1.0]. proportion of the dataset validate (only on training)
        :param path_to_model: the path to my model file
        :param name: my model file name
        :param gpu: use gpu or not (default is not)
        :param batch_size: the batch size
        :param epochs: #of epochs
        """

        self.data_path = data_path
        self.original_file_list = original_file_list
        self.image_size = image_size
        self.picture_suffix = picture_suffix

        if not isinstance(picture_suffix, list):
            self.picture_suffix = [picture_suffix]

        self.test_rate = test_rate
        self.valid_rate = valid_rate
        self.test_data = None

        self.path = path_to_model
        self.name = name
        self.gpu = gpu
        self.model = None
        self.data_iterator = None

        self.epochs = epochs
        self.batch_size = batch_size

        if self.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            os.environ['KERAS_BACKEND'] = 'tensorflow'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def load_data(self):
        """
        Load data for model (set self.data_iterator)
        """
        self.data_iterator = MyDataIterator(data_path=self.data_path,
                                            image_data_generator=None,
                                            original_file_list=self.original_file_list,
                                            batch_size=self.batch_size,
                                            picture_suffix=self.picture_suffix,
                                            out_image_size=self.image_size,
                                            should_clean_noise=True,
                                            use_canny=True,
                                            shuffle=True
                                            )

    def compile_model(self):
        mkdir(self.path)
        self.model = Sequential()

        input_shape = (self.image_size, self.image_size, 1)
        # channels_first
        self.model.add(Conv2D(16, (3, 3), data_format='channels_last', input_shape=input_shape))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(32, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        # flatten the input
        self.model.add(Flatten(data_format='channels_last'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(units=64, activation='relu', kernel_regularizer='l2'))
        self.model.add(Dense(units=20, activation='relu', kernel_regularizer='l2'))

        # output is 6DoF
        self.model.add(Dense(units=6, activation='linear'))

        print(self.model.summary())

        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def my_load(self):
        self.model = load_model(self.get_full_path())

    def my_save(self):
        mkdir(self.path)
        save_model(self.model, self.get_full_path())
        self.model.save_weights(self.get_full_path() + '.wh')
        model_dump(self.get_full_path() + '.pkl', self.model)

    def get_full_path(self):
        count_files_in_dir(self.path, self.model)
        return self.path + '/' + self.name

    def train_model(self, image_data_generator=None, save_dir=None, plot=False):
        self.data_iterator.image_generator = image_data_generator
        self.data_iterator.set_gen_labels(True, save_dir)

        test_files, validation_files = self.data_iterator.split_to_train_and_validation(self.test_rate, self.valid_rate)

        self.test_data = MyDataIterator(self.data_path,
                                        original_file_list=test_files,
                                        out_image_size=self.image_size,
                                        batch_size=self.batch_size)

        validation_data = MyDataIterator(self.data_path,
                                         original_file_list=test_files,
                                         out_image_size=self.image_size,
                                         batch_size=self.batch_size)

        callback_list = [EarlyStopping(monitor='val_loss', patience=25),
                         ModelCheckpoint(self.get_full_path() + 'best_model.h5', monitor='val_loss', mode='min',
                                         save_best_only=True)]

        hist = self.model.fit_generator(generator=self.data_iterator,
                                        validation_data=validation_data,
                                        callbacks=callback_list,
                                        epochs=self.epochs,
                                        use_multiprocessing=False,
                                        verbose=2
                                        )

        # print()
        print('Train loss:', self.model.evaluate_generator(self.data_iterator, use_multiprocessing=False))
        print('  Val loss:', self.model.evaluate_generator(validation_data, use_multiprocessing=False))
        print(' Test loss:', self.model.evaluate_generator(self.test_data, use_multiprocessing=False))

        if plot:
            history = hist.history
            loss_train = history['loss']
            loss_val = history['val_loss']

            plt.figure()
            plt.plot(loss_train, label='train')
            plt.plot(loss_val, label='val_loss', color='red')
            plt.legend()

    def model_predict(self, plot=False, save_score=True):
        if self.test_data is None:
            self.test_data = self.data_iterator

        y_pred = self.model.predict_generator(self.test_data, verbose=1)
        print(">>>>> y_pred: " + str(y_pred))
        # diff = self.data.y_test_set - y_pred

        # if plot:
        #     diff_roll = diff[:, 0]
        #     diff_pitch = diff[:, 1]
        #     diff_yaw = diff[:, 2]
        #
        #     plt.figure(figsize=(16, 10))
        #
        #     plt.subplot(3, 1, 1)
        #     plt.plot(diff_roll, color='red')
        #     plt.title('roll')
        #
        #     plt.subplot(3, 1, 2)
        #     plt.plot(diff_pitch, color='red')
        #     plt.title('pitch')
        #
        #     plt.subplot(3, 1, 3)
        #     plt.plot(diff_yaw, color='red')
        #     plt.title('yaw')
        #
        #     plt.tight_layout()
        #     print('blabla')
        #
        # if save_score:
        #     mkdir(self.path)
        #     csv_data = []
        #     for index in range(len(y_pred)):
        #         csv_data.append()
        #     write_csv()
