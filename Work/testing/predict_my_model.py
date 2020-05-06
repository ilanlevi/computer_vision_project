import time

import matplotlib.pyplot as plt

from compare_utils import compare_scores, plot_diff, plot_diff_each_param
from consts import DOWNLOAD_FOLDER, VALIDATION_CSV_2
from my_models import MyModel

PREDICT = False
if __name__ == '__main__':
    if PREDICT:
        model = MyModel(data_path=DOWNLOAD_FOLDER,
                        picture_suffix=['.png', '.jpg'],
                        gpu=True,
                        name='MyFirstTry')

        start = time.time()
        model.my_load('../model_values/MyFirstTry.pkl')
        print('> model loading completed! (in: %.2f seconds)' % (time.time() - start))

        start = time.time()
        model.model_predict(plot=True)
        print('> model prediction completed! (in: %.2f seconds)' % (time.time() - start))

    validation_folder = 'C:/Work/ComputerVision/valid_set/New folder/validset_2/'
    diff_file = 'diff.csv'
    my_file = 'scores.csv'
    compare_scores(validation_folder, VALIDATION_CSV_2, my_file, diff_file, False)
    plot_diff(validation_folder, diff_file, title='diff')

    plot_diff_each_param(validation_folder, [VALIDATION_CSV_2, my_file])

    plt.show()
