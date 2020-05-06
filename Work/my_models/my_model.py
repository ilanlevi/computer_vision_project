import os

from keras import applications, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Dense, Flatten
from keras.models import save_model
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

from consts import DEFAULT_VALID_RATE, MY_MODEL_LOCAL_PATH, MY_MODEL_NAME, BATCH_SIZE, EPOCHS, \
    PICTURE_SUFFIX, PICTURE_SIZE, DEFAULT_TEST_RATE, OUT_DIM, FINAL_OUTPUT_CSV_VALUES_LABELS
from image_utils import FpnWrapper
from my_models import ImagePoseGenerator
from my_utils import my_mkdir, model_dump, model_load, write_csv, get_suffix


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

        self.base_model = None
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
        """
        Define model architecture and compile it
        """
        # create dir
        my_mkdir(self.path)
        # define input size
        input_shape = (self.image_size, self.image_size, 3)
        data_format = 'channels_last'

        self.base_model = applications.ResNet101V2(include_top=False, input_shape=input_shape)

        # append my model to ResNet101
        x = self.base_model.output
        x = Flatten(data_format=data_format)(x)
        # output is 3DoF
        predictions = Dense(units=OUT_DIM, activation='linear')(x)
        self.model = Model(inputs=self.base_model.input, outputs=predictions)

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        print(self.model.summary())

    def my_load(self, file):
        """
        Load model from .pkl file
        :param file: the full file path
        """
        self.model = model_load(file)

    def keras_load(self, file):
        """
        Same as my_load, just with .h5 file (keras format)
        :param file: the full file path
        """
        self.model = load_model(file)

    def my_save(self, pkl_format=True, keras_format=True):
        """
        Save model values to self.path with self.name files
        :param pkl_format: save to pkl file ot not
        :param keras_format: save to h5 file ot not
        """
        # create dir if doesn't exist
        my_mkdir(self.path)
        if keras_format:
            # save to keras
            save_model(self.model, self.get_full_path() + '.h5')
        if pkl_format:
            # save pkl file
            model_dump(self.get_full_path() + '.pkl', self.model)

    def get_full_path(self):
        """
        return full path - directory + model name
        :return: new full file path
        """
        return self.path + '/' + self.name

    def train_model(self, datagen_args):
        """
        Train model (after compilation)
        :param datagen_args: the keras image generation params
        :return: histogram, poseGenerator, validation_data of training
        """
        # define input images size
        INPUT_SIZE = (self.image_size, self.image_size)

        # create keras image generation for images - that how i'm doing the face augmentation
        data_gen = ImageDataGenerator(**datagen_args)

        # create my pose generation with face augmentation settings
        poseGenerator = ImagePoseGenerator(self.data_path,
                                           image_data_generator=data_gen,
                                           image_shape=INPUT_SIZE,
                                           color_mode='rgb',
                                           batch_size=self.batch_size,
                                           gen_y=True,
                                           to_gray=False,
                                           shuffle=True,
                                           follow_links=True,
                                           d_type='float32',
                                           )

        validation_files, _ = poseGenerator.split_to_train_and_validation(DEFAULT_TEST_RATE)

        # define the validation set (with no augmentation)
        validation_data = ImagePoseGenerator(self.data_path,
                                             to_gray=False,
                                             f_list=validation_files,
                                             color_mode='rgb',
                                             gen_y=False,
                                             shuffle=True,
                                             image_shape=INPUT_SIZE,
                                             batch_size=5,
                                             follow_links=True,
                                             d_type='float32',
                                             )

        best_model_name = self.get_full_path() + '_best_model.h5'

        # define early stopping to prevent over-fitting
        # also, save checkpoint on best values
        callback_list = [
            EarlyStopping(monitor='val_loss', patience=50),
            ModelCheckpoint(best_model_name, monitor='val_loss', mode='min', save_best_only=True, period=5)
        ]

        steps_per_epoch = len(poseGenerator)
        print('steps_per_epoch: %d' % steps_per_epoch)

        # run training and save histogram
        hist = self.model.fit_generator(poseGenerator,
                                        validation_data=validation_data,
                                        callbacks=callback_list,
                                        validation_steps=101,
                                        epochs=self.epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        verbose=1,
                                        workers=5,
                                        use_multiprocessing=False)

        # save model
        self.my_save()

        return hist, poseGenerator, validation_data

    def eval_model(self, hist, poseGenerator, validation_data, plot=False):
        """
        Evaluates the model performance on the selected data
        :param hist: the training histogram
        :param poseGenerator: face augmentation generator
        :param validation_data: the validation set data
        :param plot: to plot data or not
        """
        print('Train loss:', self.model.evaluate_generator(poseGenerator, use_multiprocessing=False, workers=1))
        print('  Val loss:', self.model.evaluate_generator(validation_data, use_multiprocessing=False, workers=1))

        if plot:
            history = hist.history
            loss_train = history['loss']
            loss_val = history['val_loss']
            accuracy = history['accuracy']

            plt.figure()
            plt.plot(loss_train, label='train', color='green')
            plt.plot(loss_val, label='val_loss', color='red')
            plt.plot(accuracy, label='accuracy', color='blue')

            plt.legend()

    def model_predict(self, file_list, output_file_full_path, datagen_args=None, fpn_wrapper=FpnWrapper()):
        """
        Run model prediction on the files from file_list
        :param fpn_wrapper: the fpn wrapper
        :param datagen_args: the datagen_args
        :param file_list: the files list to run prediction on
        :param output_file_full_path: the output csv file to write scores
        """
        if datagen_args is None:
            datagen_args = dict()

        INPUT_SIZE = (self.image_size, self.image_size)
        data_gen = ImageDataGenerator(**datagen_args)

        poseGenerator = ImagePoseGenerator(os.getcwd(),
                                           image_data_generator=data_gen,
                                           fpn_model=fpn_wrapper,
                                           image_shape=INPUT_SIZE,
                                           color_mode='rgb',
                                           batch_size=1,
                                           gen_y=False,
                                           to_gray=False,
                                           shuffle=False,
                                           follow_links=False,
                                           d_type='float32',
                                           f_list=file_list
                                           )

        scores = []
        for i in range(len(file_list)):
            [[rx, ry, rz]] = self.model.predict(poseGenerator.next(), verbose=2)
            f_name = get_suffix(file_list[i], '/')
            if f_name is None or len(f_name) is 0:
                f_name = file_list[i]

            s = [f_name, rx, ry, rz]

            scores.append(s)

        write_csv(scores, FINAL_OUTPUT_CSV_VALUES_LABELS, output_file_full_path, '',
                  append=False, print_data=True)
