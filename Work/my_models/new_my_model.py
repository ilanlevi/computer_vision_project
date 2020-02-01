import os

import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from keras.models import load_model, save_model, Sequential
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from consts import DEFAULT_VALID_RATE, MY_MODEL_LOCAL_PATH, MY_MODEL_NAME, BATCH_SIZE, EPOCHS, \
    PICTURE_SUFFIX, PICTURE_SIZE, DEFAULT_TEST_RATE, VALIDATION_CSV_2, PICTURE_NAME, CSV_VALUES_LABELS
from my_models import ImagePoseGenerator
from my_utils import my_mkdir, model_dump, count_files_in_dir


class MyNewModel:

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

    def compile_model(self):
        my_mkdir(self.path)
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
        my_mkdir(self.path)
        save_model(self.model, self.get_full_path())
        self.model.save_weights(self.get_full_path() + '.wh')
        model_dump(self.get_full_path() + '.pkl', self.model)

    def get_full_path(self):
        count_files_in_dir(self.path, self.model)
        return self.path + '/' + self.name

    def train_model(self, datagen_args, save_dir=None, plot=False):
        INPUT_SIZE = (self.image_size, self.image_size)
        seed = 1

        # image_datagen = ImageDataGenerator(**datagen_args)
        masks_datagen = ImageDataGenerator(**datagen_args)
        validation_datagen = ImageDataGenerator(**datagen_args)


        # image_generator = image_datagen.flow_from_directory(
        #     self.data_path,
        #     class_mode=None,
        #     target_size=INPUT_SIZE,
        #     color_mode='grayscale',
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     follow_links=True,
        #     seed=seed)

        pose_datagen = ImagePoseGenerator(self.data_path,
                                          masks_datagen,
                                          mask_size=INPUT_SIZE,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          follow_links=True,
                                          seed=seed
                                          )

        validation_folder = 'C:\\Work\\ComputerVision\\valid_set\\validset_2\\'
        validation_file = validation_folder + VALIDATION_CSV_2
        df = pd.read_csv(validation_file)
        df = df.astype({'rx': float, 'ry': float, 'rz': float, 'tx': float, 'ty': float, 'tz': float})

        validation_data = validation_datagen.flow_from_dataframe(dataframe=df,
                                                                 directory=validation_folder,
                                                                 x_col=PICTURE_NAME,
                                                                 y_col=CSV_VALUES_LABELS,
                                                                 class_mode="raw",
                                                                 shuffle=True,
                                                                 color_mode='grayscale',
                                                                 target_size=INPUT_SIZE,
                                                                 batch_size=self.batch_size)

        callback_list = [EarlyStopping(monitor='val_loss', patience=25),
                         ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min',
                                         save_best_only=True)]

        # images_and_pose = zip(image_generator, pose_datagen)
        steps_per_epoch = len(pose_datagen) // self.batch_size
        # masks_datagen.fit(pose_datagen.next()[0], seed=seed)

        hist = self.model.fit_generator(pose_datagen,
                                        validation_data=validation_data,
                                        callbacks=callback_list,
                                        epochs=self.epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        verbose=1,
                                        workers=5,
                                        use_multiprocessing=False)

        # print('asdasd')
        print('Train loss:', self.model.evaluate_generator(pose_datagen, use_multiprocessing=False, workers=5))
        print('  Val loss:', self.model.evaluate_generator(validation_data, use_multiprocessing=False, workers=5))
        # print(' Test loss:', self.model.evaluate_generator(self.test_data, use_multiprocessing=True, workers=5))

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
