import time

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
        im = load_image(image_path, size, gray, print_data)
        if im is not None:
            ims.append(im)

    if print_data:
        print('> Loading images took: %.2f seconds' % (time.time() - start))

    return ims


def load_image(image_path, size=None, gray=True, print_data=False):
    """
    load image: resize image and convert to gray scale
    :param print_data: print duration data (default is false)
    :param gray: convert to gray or not
    :param size: size of image - image will be sized: (size x size)
    :param image_path: images list (full path) to load
    :return: image
    """

    start = time.time()
    im = None
    try:
        if gray:
            im = cv2.imread(image_path, 0)
            # # todo - remove
            # im = cv2.flip(im, 1)
        else:
            im = cv2.imread(image_path)
        im = my_resize(im, size, size)
    except Exception as e:
        print('Error while reading image!Path= %s\nError= %s' % (image_path, str(e)))

    if print_data:
        print('Loading image took: %.2f seconds' % (time.time() - start))

    return im


def my_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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
