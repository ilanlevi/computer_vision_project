import time

import numpy as np
from sklearn.decomposition import PCA

from Work.data.mnist_data import MNIST
from abstract_classifier import AbstractClassifier


class MyPCA(AbstractClassifier):
    DEFAULT_STEP_SIZE_KP = 30
    DEFAULT_N_COMPONENTS = 0

    def __init__(self, n_components=DEFAULT_N_COMPONENTS, data_reader=None):
        super(MyPCA, self).__init__(n_components, data_reader)
        self.pca = None  # type: PCA
        self.data_reader = data_reader  # type: MNIST
        self.step_size = MyPCA.DEFAULT_STEP_SIZE_KP

    def projected_data(self, n_dimensions, x_data, pca=None, fit=True):
        """
        Transform the data using pca (self or from param),
        and get the first n_dimensions components
        :param n_dimensions: number of dimensions (components)
        :param x_data: the original data
        :param pca: the fitted pca (in none-> from self)
        :param fit: fit and transform or just transform the data
        :return: list of components
        """
        if pca is None:
            pca = self.pca

        if fit:
            pca_projected_components = pca.fit_transform(x_data)
        else:
            pca_projected_components = pca.transform(x_data)

        components = [pca_projected_components[:, i] for i in range(n_dimensions)]

        return components

    def approximate_data(self, n_dimensions, x_data=None, pca=None, fit=True):
        """
        Use the pca transform on data and then do an inverse transform for approximation
        :param n_dimensions: number of dimensions
        :param x_data: the data (if None -> self.data_reader.x_train)
        :param pca: the fitted pca (in none-> from self)
        :param fit: fit and transform or just transform the data
        :return: a tuple of: (approximation data, projected data)
        """
        if pca is None:
            pca = self.pca

        projected = self.projected_data(n_dimensions, x_data, pca, fit)
        projected = np.asarray(projected)
        projected_t = projected.T
        approximation = pca.inverse_transform(projected_t)
        return approximation, projected

    # abstract
    def train(self, data=None, labels=None, descriptors=None):
        start = time.time()
        if data is None:
            data = self.data_reader.x_train_set

        # train pca
        if self.n_components <= MyPCA.DEFAULT_N_COMPONENTS:
            self.pca = PCA()
        else:
            self.pca = PCA(n_components=self.n_components)

        self.pca.fit(data)

        end = time.time()
        print ('Total training took: %.2f sec.' % (end - start))

        return self

    def test_score(self, test_data=None, test_labels=None):
        """
            Does nothing
        """
        return self
