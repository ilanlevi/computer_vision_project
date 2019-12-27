import time

import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from Work import tools as pre_data
from tmp.data import MNIST
from abstract_classifier import AbstractClassifier


class KNN(AbstractClassifier):
    DEFAULT_STEP_SIZE_KP = 15
    DEFAULT_K_CLUSTERS = 1
    DEFAULT_N_COMPONENTS = 50

    def __init__(self, n_components=DEFAULT_N_COMPONENTS, data_reader=None, k_clusters=DEFAULT_K_CLUSTERS,
                 step_size=DEFAULT_STEP_SIZE_KP):
        super(KNN, self).__init__(n_components, data_reader)
        self.k_clusters = k_clusters
        self.knn = None  # type = KNeighborsClassifier
        self.data_reader = data_reader  # type: MNIST
        self.step_size = step_size

    def create_descriptors(self, data):
        """
        Create Dense-SIFT for data with param from self on data
        :param data - matrix (samples X sample_size)
        :return: transformed descriptors matrix [samples X descriptors_size]
        """
        descriptors = pre_data.generate_dsift(data, self.data_reader.get_picture_size(), self.step_size,
                                              self.n_components, False)
        descriptors = np.reshape(descriptors, (descriptors.shape[0], descriptors.shape[1] * descriptors.shape[2]))
        print ('descriptors shape: %s' % (str(np.shape(descriptors))))

        # standardize
        descriptors = StandardScaler().fit_transform(descriptors)

        return descriptors

    # abstract
    def train(self, data=None, labels=None, descriptors=None):
        start = time.time()
        if data is None:
            data = self.data_reader.x_train_set
        if labels is None:
            labels = self.data_reader.y_train_set

        # calculate Dense-SIFT
        if descriptors is None:
            descriptors = self.create_descriptors(data)

        # perform clustering
        self.fit_knn(self.k_clusters, descriptors, labels)

        end = time.time()
        print ('Total training took: %.2f sec.' % (end - start))

        return self

    def fit_knn(self, k_clusters, features, features_labels):
        """
            set KNN for self.knn with fitting features in k_clusters
            :param features_labels: features labels
            :param k_clusters: the number of centers
            :param features: the images features
            :return: self
        """
        start = time.time()
        self.knn = KNeighborsClassifier(n_neighbors=k_clusters, )
        self.knn.fit(features, features_labels)
        end = time.time()
        print ('KNN fitting [k = %d] took: %.2f sec.' % (k_clusters, end - start))

        return self

    def test_score(self, test_data=None, test_labels=None, descriptors=None):
        start = time.time()

        if test_data is None:
            test_data = self.data_reader.x_test_set
        if test_labels is None:
            test_labels = self.data_reader.y_test_set

        # calculate Dense SIFT if nedded
        if descriptors is None:
            descriptors = self.create_descriptors(test_data)

        predicted = self.knn.predict(descriptors)

        predict_score = metrics.accuracy_score(test_labels, predicted)

        end = time.time()
        print ('Total testing took: %.2f sec.\n###\nScore [k = %d] = %.3f' % (
            end - start, self.k_clusters, predict_score))

        return predict_score
