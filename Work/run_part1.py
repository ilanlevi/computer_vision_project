import numpy as np

from Work.data.mnist_data import MNIST
from Work.show_data.images_as_sub_plots import ImagesGraph

PATH_TO_MNIST = "C:\\Users\\ilan_\\Google Drive\\Masters\\Computer vision\\Assignments\\12\\Work\\"
MNIST_NAME = 'mnist.gz'
NUMBER_OF_DIGITS = 10
NUMBER_OF_DIGITS_TO_SHOW = 12
N_COL = 4

if __name__ == '__main__':
    mnist = MNIST(data_path=PATH_TO_MNIST, file_name=MNIST_NAME).init()

    # show how num of digits
    dig_array = mnist.count_digit()
    for index in range(NUMBER_OF_DIGITS):
        print('digit = %d -> %d times in set!' % (index, dig_array[index]))

    # show first pictures
    dig_array = np.copy(mnist.x_train_set)[:NUMBER_OF_DIGITS_TO_SHOW]
    labels = ['MNIST[%d] = %d' % (index, mnist.y_train_set[index]) for index in range(NUMBER_OF_DIGITS_TO_SHOW)]
    s = mnist.get_picture_size()
    dig_array = np.reshape(dig_array, (dig_array.shape[0], s, s))

    g = ImagesGraph(col_size=N_COL)
    g.show_data(images=dig_array, images_labels=labels, max_images=NUMBER_OF_DIGITS_TO_SHOW)
