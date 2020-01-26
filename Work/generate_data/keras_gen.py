import numpy as np

from consts import DataFileConsts as dConst, FPNConsts
from image_utils import load_image
from models import load_fpn_model
from mytools import get_landmarks


# todo - delete
def gen_lmarks_and_image(image_path):
    image = load_image(image_path, gray=False)
    shape = image.shape
    lmarks_image = np.zeros((shape[0], shape[1]))
    lmarks = get_landmarks(image_path)
    lmarks = np.asarray(lmarks)
    for p in lmarks:
        lmarks_image[int(p[1]), int(p[0])] = 1

    return image, lmarks_image, lmarks


def load_models():
    # load model
    model_path = FPNConsts.THIS_PATH + FPNConsts.MODELS_DIR
    model_file_name = FPNConsts.POSE_P
    model_name = FPNConsts.MODEL_NAME

    camera_matrix, model_matrix = load_fpn_model(model_path, model_file_name, model_name)

    return camera_matrix, model_matrix


def add(x):
    x = x + '1'
    return x


if __name__ == '__main__':
    folder = dConst.DOWNLOAD_FOLDER
    output_folder = dConst.OUTPUT_FOLDER

    output_path = 'C:/Work/ComputerVision/datasets/DATA/outmy'

    suffixes = [data_set['IMAGE_SUFFIX'] for data_set in dConst.ALL_DATASETS]
    suffixes = np.asarray(suffixes)
    suffixes = np.unique(suffixes, axis=0)

    files_list = []

    input_file_list = ['C:\\Work\\ComputerVision\\datasets\\DATA\\lfpw\\testset\\image_0001.png']

    i, l_i, _ = gen_lmarks_and_image(input_file_list[0])
    cam, model = load_models()
    y = zip('xxx', 'yyy')
    print("Before: " + str(y))
    y = add(y)
    print("After: " + str(y))
    #
    #
    #
    # data_gen_args = dict(featurewise_center=True,
    #                      featurewise_std_normalization=True,
    #                      rotation_range=40,
    #                      width_shift_range=0.2,
    #                      height_shift_range=0.2,
    #                      horizontal_flip=True,
    #                      zoom_range=0.2)
    # image_datagen = ImageDataGenerator(**data_gen_args)
    # mask_datagen = ImageDataGenerator(**data_gen_args)
    # # Provide the same seed and keyword arguments to the fit and flow methods
    # seed = 1
    # image_datagen.fit(i, augment=True, seed=seed)
    # mask_datagen.fit(l_i, augment=True, seed=seed)
    # image_generator = image_datagen.flow_from_directory(
    #     'data/images',
    #     class_mode=None,
    #     seed=seed)
    # mask_generator = mask_datagen.flow_from_directory(
    #     'data/masks',
    #     class_mode=None,
    #     seed=seed)
    #
    # # combine generators into one which yields image and masks
    # train_generator = zip(image_generator, mask_generator)
    # model.fit_generator(
    #     train_generator,
    #     steps_per_epoch=2000,
    #     epochs=50)
