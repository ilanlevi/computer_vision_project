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
    salt_vs_pepper_ratio = 0.5
    amount = 0.0005

    salted_and_peppered = image.copy()
    mean = salted_and_peppered.mean()

    num_salt = np.ceil(amount * image.size * salt_vs_pepper_ratio)
    num_pepper = np.ceil(amount * image.size * (1 - salt_vs_pepper_ratio))

    # add salt
    xy_s = [np.random.randint(0, i - 1, int(num_salt)) for i in salted_and_peppered.shape]
    salted_and_peppered[xy_s[0], xy_s[1]] = 1 * mean

    # add pepper
    xy_s = [np.random.randint(0, i - 1, int(num_pepper)) for i in salted_and_peppered.shape]
    salted_and_peppered[xy_s[0], xy_s[1]] = 0

    return salted_and_peppered


def _poisson(image):
    """
    Add poisson noise to image
    :param image: the image
    :return: noisy image
    """
    values = 2 ** np.ceil(np.log2(len(np.unique(image))))
    noisy = np.random.poisson(image * values) / float(values)

    return noisy
