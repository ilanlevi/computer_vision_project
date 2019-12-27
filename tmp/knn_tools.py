# import time
#
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from matplotlib import pyplot as plt
#
# # from Work import tools as pre_data
# # from Work.data import MNIST
# # from Work.show_data import PlotsGraph
# # from Work.classification import KNN
#
#
# class KNNTools:
#
#     def __init__(self, k_max=1, mnist=None):
#         self.k_max = k_max
#         self.mnist = mnist  # type: MNIST
#
#     def generate_descriptors(self, train_set=None, test_set=None, picture_size=None):
#         """
#         Create Dense-SIFT for data
#         :param picture_size: image size
#         :param test_set: the test set array
#         :param train_set: the train set array
#         :return: transformed descriptors matrix [samples X descriptors_size]
#         """
#         if train_set is None:
#             train_set = self.mnist.x_train_set
#         if test_set is None:
#             test_set = self.mnist.x_test_set
#         if picture_size is None:
#             picture_size = self.mnist.get_picture_size()
#         des = pre_data.generate_dsift(train_set, picture_size, KNN.DEFAULT_STEP_SIZE_KP,
#                                       KNN.DEFAULT_N_COMPONENTS, False)
#         des_test = pre_data.generate_dsift(test_set, picture_size,
#                                            KNN.DEFAULT_STEP_SIZE_KP,
#                                            KNN.DEFAULT_N_COMPONENTS, False)
#         # calculate Dense-SIFT
#
#         des = np.reshape(des, (des.shape[0], des.shape[1] * des.shape[2]))
#         des_test = np.reshape(des_test, (des_test.shape[0], des_test.shape[1] * des_test.shape[2]))
#
#         print ('descriptors shape: %s' % (str(np.shape(des))))
#         print ('descriptors-test shape: %s' % (str(np.shape(des_test))))
#
#         # standardize
#         des = StandardScaler().fit_transform(des)
#         des_test = StandardScaler().fit_transform(des_test)
#
#         return des, des_test
#
#     def test_knn(self, k_value, descriptors, descriptors_test):
#         """
#         Create and test knn scores
#         :param k_value: the number of clusters
#         :param descriptors: the train descriptors
#         :param descriptors_test: the test set descriptors
#         :return: score
#         """
#         s = time.time()
#         # create new knn
#         knn = KNN(data_reader=self.mnist, k_clusters=k_value)
#         # train knn
#         knn.train(descriptors=descriptors, labels=self.mnist.y_train_set)
#         # evaluate knn
#         score = knn.test_score(descriptors=descriptors_test, test_labels=self.mnist.y_test_set)
#         e = time.time()
#         # save score
#
#         print("k=%d, accuracy=%.2f%%" % (k_value, score * 100))
#         print("Complete time: %.2f Secs.\n" % (e - s))
#
#         return score
#
#     def test_knn_range(self, des=None, des_test=None, title='KNN scores', plt_wait=False):
#         if des is None or des_test is None:
#             des, des_test = self.generate_descriptors()
#
#         scores = []
#         k_s = range(1, self.k_max)
#         for k in k_s:
#             score = self.test_knn(k, des, des_test)
#             # save score
#             scores.append(score)
#
#         # find the value of k that has the largest accuracy
#         i = int(np.argmax(scores))
#         print("!@#!@#!@\nK = %d achieved highest accuracy of %.3f on validation data" % (k_s[i], scores[i]))
#
#         # show graph
#
#         g = PlotsGraph(title=title)
#         g.show_data(k_s, np.asarray(scores), 'K value', 'Accuracy')
#
#         if plt_wait:
#             plt.show()
