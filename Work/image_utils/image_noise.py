import numpy as np

NOISES = [
    lambda x: x,  # return original image
    lambda x: _gauss(x),
    lambda x: _salt_and_pepper(x),
    lambda x: _poisson(x)
]


def random_noisy(image):
    """
    Run a random pick of noise on image
    :param image: the image
    :return: noisy or original image
    """
    noise_type_index = np.random.randint(0, len(NOISES))
    new_image = NOISES[noise_type_index](image)
    return new_image


def _gauss(image):
    """
    Add gauss noise to image
    :param image: the image
    :return: noisy image
    """
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    gauss = np.random.normal(mean, sigma, image.shape)
    gauss = gauss.reshape(image.shape)
    noisy = image + gauss

    return noisy


def _salt_and_pepper(image):
    """
    Add salt and pepper noise to image
    :param image: the image
    :return: noisy image
    """
    s_vs_p = 0.5
    amount = 0.004

    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    xy_s = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
    out[xy_s] = 1

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    xy_s = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
    out[xy_s] = 0

    return out


def _poisson(image):
    """
    Add poisson noise to image
    :param image: the image
    :return: noisy image
    """
    values = 2 ** np.ceil(np.log2(len(np.unique(image))))
    noisy = np.random.poisson(image * values) / float(values)

    return noisy
