import cv2
import time


class ImageTools:

    def __init__(self):
        pass

    @staticmethod
    def load_images(images_path_list, width):
        start = time.time()
        ims = []
        for image_path in images_path_list:
            try:
                im = cv2.imread(image_path)
                im = ImageTools.resize(im, width=width)
                ims.append(im)
            except Exception as e:
                print ('Error while reading image!Path= %s\nError= %s' % (image_path, str(e)))

        print 'Loading images took: %.2f seconds' % (time.time() - start)
        return ims

    @staticmethod
    def load_converted_images(images_path_list, width):
        start = time.time()
        ims = []
        for image_path in images_path_list:
            try:
                im = cv2.imread(image_path)
                im = ImageTools.convert_image(im, width)
                ims.append(im)
            except Exception as e:
                print ('Error while reading image!Path= %s\nError= %s' % (image_path, str(e)))

        print 'Loading and converting images took: %.2f seconds' % (time.time() - start)
        return ims

    @staticmethod
    def convert_images(images, width=500):
        """
        for each image: resize image and convert to gray scale
        :param width: the width
        :param images: the loaded image
        :return: gray scaled image list
        """
        gray_images = []
        for im in images:
            gray = ImageTools.convert_image(im, width)
            gray.append(gray)
        return gray_images

    @staticmethod
    def convert_image(image, width=500):
        """
        resize image and convert to gray scale
        :param width: the width
        :param image: the loaded image
        :return: gray scaled image
        """
        im = ImageTools.resize(image, width=width)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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
