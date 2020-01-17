import time

import numpy as np
import cv2


def load_images(images_path_list, size=None, gray=True, print_data=False):
    """
    for each image: resize image and convert to gray scale
    :param print_data: print duration data (default is false)
    :param gray: convert to gray or not
    :param size: size of image - image will be sized: (size x size)
    :param images_path_list: images list (full path) to load
    :return: image list
    """

    start = time.time()

    ims = []
    for image_path in images_path_list:
        try:
            if gray:
                im = cv2.imread(image_path, 0)
            else:
                im = cv2.imread(image_path)
            im = resize(im, size, size)
            ims.append(im)
        except Exception as e:
            print ('Error while reading image!Path= %s\nError= %s' % (image_path, str(e)))

    if print_data:
        print 'Loading images took: %.2f seconds' % (time.time() - start)

    return ims


def save_images(images, path, print_data=False):
    """
    for each image: save image with name
    :param print_data: print duration data (default is false)
    :param images: images list of tuples: (image, name)
    :param path: images path to save (directory)
    :return: None
    """

    start = time.time()

    ims = []
    for name, image in images:
        try:
            full_path = path + name
            cv2.imwrite(full_path, image)
        except Exception as e:
            if print_data:
                print ('Error while saving image!Path= %s\nError= %s' % (name, str(e)))

    if print_data:
        print 'Saving images took: %.2f seconds' % (time.time() - start)

    return ims


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Resize image
    :param image: the image
    :param width: new width
    :param height: new height
    :param inter: cv2 interpolation
    :return: the resized image
    """

    if width is None or height is None:
        return image

    # resize the image
    resized = cv2.resize(image, (width, height), interpolation=inter)

    # return the resized image
    return resized


def resize_image_and_landmarks(image, landmarks, new_size=None, inter=cv2.INTER_AREA):
    """
    Resize image
    :param image: the image
    :param landmarks: the image landmarks points
    :param new_size: new squared image shape
    :param inter: cv2 interpolation
    :return: tuple of: (resized_image, resized_landmarks)
    """

    if new_size is None:
        return image, landmarks

    # get ratio
    original_shape = image.shape
    ratio_x, ratio_y = (new_size / float(original_shape[0])), (new_size / float(original_shape[1]))

    # resize the image
    resized = cv2.resize(image, (new_size, new_size), interpolation=inter)

    # resize landmarks
    resized_landmarks = np.array(landmarks)
    resized_landmarks[:, 0] = resized_landmarks[:, 0] * ratio_y
    resized_landmarks[:, 1] = resized_landmarks[:, 1] * ratio_x

    # return the resized image
    return resized, resized_landmarks
