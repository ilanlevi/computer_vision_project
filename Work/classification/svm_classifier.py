import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC

import Work.tools.prepossess_data as pre_data
from Work.classification.bow import Bow
from Work.data.spatial_data import Spatial
from abstract_classifier import AbstractClassifier


class SVM(AbstractClassifier):
    DEFAULT_STEP_SIZE_KP = 30
    DEFAULT_K_CLUSTERS = 150
    DEFAULT_N_COMPONENTS = 150
    DEFAULT_C_VALUE = 0.1

    def __init__(self, c_value=DEFAULT_C_VALUE, n_components=DEFAULT_N_COMPONENTS, data_reader=None):
        super(SVM, self).__init__(n_components, data_reader)
        self.c_value = c_value
        self.k_means = None
        self.data_reader = data_reader  # type: Spatial
        self.svm = None
        self.step_size_kp = SVM.DEFAULT_STEP_SIZE_KP
        self.k_clusters = SVM.DEFAULT_K_CLUSTERS

    # abstract
    def train(self, c_value=None, data=None, labels=None, features=None, use_learning=False):
        start = time.time()
        if data is None:
            data = self.data_reader.x_train_set
        if labels is None:
            labels = self.data_reader.y_train_set
        if c_value is None:
            c_value = self.c_value

        if features is None:
            # if use_learning:
            # calculate features with learning
            # features = pre_data.generate_learning_features(data, self.data_reader.get_picture_size())
            # else:
            # calculate dense-SIFT if needed
            # features = pre_data.generate_dsift(data, self.data_reader.get_picture_size(), self.step_size_kp,
            #                                    self.n_components)
            features = pre_data.generate_dsift(data, self.data_reader.get_picture_size(), self.step_size_kp,
                                               self.n_components)
        print ('features shape: %s' % (str(np.shape(features))))

        # perform clustering
        bow = Bow(n_clusters=self.n_components)  # init bag of word
        b_stack = bow.format_nd(features)

        predict_cluster = self.fit_k_mean(self.k_clusters, b_stack)
        print ('predict_cluster shape: %s' % (str(np.shape(predict_cluster))))

        # create bag of words
        bow.generate_vocabulary(n_images=len(features), descriptor_list=features, k_means_returned=predict_cluster)
        bow.standardize()
        vocabulary = bow.vocabulary

        # trains svm
        self.train_svm(c_value, vocabulary, labels)

        end = time.time()
        print ('Total training took: %.2f sec.' % (end - start))

        return self

    def fit_k_mean(self, k_clusters, features):
        """
            set K_MEANS for self.k_means with fitting features in k_clusters
            :param k_clusters: the number of centers
            :param features: the images features
            :return: k-means.predict() on the features - i.e predict cluster index for each sample
        """
        start = time.time()
        self.k_means = KMeans(n_clusters=k_clusters, n_jobs=-1).fit(features)
        end = time.time()
        print ('K-Means fitting took: %.2f sec.' % (end - start))
        start = time.time()
        predict_cluster = self.k_means.predict(features)
        end = time.time()
        print ('K-Means predict took: %.2f sec.' % (end - start))
        return predict_cluster

    def train_svm(self, c_value, predict_cluster, labels):
        """
        The fit method of SVC class is called to train the algorithm on the training data.
        Set the self.svm  as fitted svm.
        :param c_value: margin size for the 2 classes
        :param predict_cluster: predict cluster index for each sample from the m-means
        :param labels: the images labels
        :return: self
        """
        start = time.time()
        self.svm = LinearSVC(C=c_value, random_state=0, tol=1e-5)
        self.svm.fit(predict_cluster, labels)
        end = time.time()
        print ('Linear SVM fitting took: %.2f sec.' % (end - start))
        return self

    def test_score(self, test_data=None, test_labels=None, features=None, use_learning=False):
        start = time.time()
        if test_data is None:
            test_data = self.data_reader.x_test_set
        if test_labels is None:
            test_labels = self.data_reader.y_test_set

        if features is None:
            # if use_learning:
            #     calculate features with learning
            #     features = pre_data.generate_learning_features(test_data, self.data_reader.get_picture_size())
            # else:
            #     calculate dense-SIFT if needed
            #     features = pre_data.generate_dsift(test_data, self.data_reader.get_picture_size(), self.step_size_kp,
            #                                        self.n_components)
            features = pre_data.generate_dsift(test_data, self.data_reader.get_picture_size(), self.step_size_kp,
                                               self.n_components)

        b = Bow(n_clusters=self.k_clusters)
        b_stack = b.format_nd(features)

        predicted = self.k_means.predict(b_stack)

        # create bag of words
        b.generate_vocabulary(n_images=len(features), descriptor_list=features, k_means_returned=predicted)
        b.standardize()
        vocabulary = b.vocabulary

        decision_scores = self.svm.predict(vocabulary)

        end = time.time()

        type1 = self.data_reader.type1_name  # self.data_reader type: Spatial
        # convert string to int array for calculating the result difference
        tests_labels = SVM.transform_string_to_int(test_labels, type1)
        score_arr = SVM.transform_string_to_int(decision_scores, type1)

        print ('Total testing took: %.2f sec.' % (end - start))

        return score_arr, tests_labels

    @staticmethod
    def transform_string_to_int(string_arr, type_name):
        """
        Map string array(type1/type2) to int array
        :param string_arr: the string array
        :param type_name: the first type
        :return: int array with 0's and 1's
        """
        ln = len(string_arr)
        a = np.zeros(ln, dtype=int)
        for index in range(ln):
            if type_name.__eq__(string_arr[index]):
                a[index] = 1
        return a
