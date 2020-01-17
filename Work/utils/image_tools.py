import time

import cv2


def load_images(images_path_list, width=None, height=None, gray=True, print_data=False):
    """
    for each image: resize image and convert to gray scale
    :param print_data: print duration data (default is false)
    :param gray: convert to gray or not
    :param height: the height
    :param images_path_list: images list (full path) to load
    :param width: the width
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
            im = resize(im, width, height)
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
            name = path + name
            cv2.imwrite(name, image)
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
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
