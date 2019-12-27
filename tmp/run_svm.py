import math

from tmp.classification.svm_classifier import SVM
from tmp.data.spatial_data import Spatial
from tmp.show_data import AccuracyPlot

PATH_TO_DATA = "C:\\Users\\ilan_\\Downloads\\Browsers\\"
ARCHIVE_NAME = 'spatial_envelope_256x256_static_8outdoorcategories (1).zip'
UN_ARCHIVE_FOLDER = 'section2data'
DATA_SET_SUB_FOLDER_NAME = 'spatial_envelope_256x256_static_8outdoorcategories'
UNZIP_FILE = True
TYPE1_NAME = 'coast'
TYPE2_NAME = 'forest'
TESTED_C_VALUES = [math.pow(0.1, i) for i in range(5)]
TRAIN_RATE = 1 / 4.0
RUN_WITH_LEARNING_FEATURES = False

if __name__ == '__main__':
    # TEST SVM

    # load data
    sp = Spatial(full_path=PATH_TO_DATA, data_path=UN_ARCHIVE_FOLDER + "\\" + DATA_SET_SUB_FOLDER_NAME,
                 target_path=UN_ARCHIVE_FOLDER, archive_path=ARCHIVE_NAME, type1_name=TYPE1_NAME, type2_name=TYPE2_NAME,
                 train_rate=TRAIN_RATE)
    sp.init(unzip_file=UNZIP_FILE)

    scores_testlbl_cvalue = []
    for c_value in TESTED_C_VALUES:
        # create svm
        p = SVM(data_reader=sp)
        # train
        p.train(use_learning=RUN_WITH_LEARNING_FEATURES)
        # save score
        test_label, score_res = p.test_score(use_learning=RUN_WITH_LEARNING_FEATURES)
        # append as a tuple
        scores_testlbl_cvalue.append((score_res, test_label, c_value))

    g = AccuracyPlot(title='SVM & BOW classification')
    g.show_data(scores_test_c_list=scores_testlbl_cvalue)
