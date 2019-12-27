from abc import ABCMeta, abstractmethod


class AbstractClassifier:
    __metaclass__ = ABCMeta

    def __init__(self, n_components=0, data_reader=None):
        """
        :type n_components: int
        :type data_reader: AbstractReadData
        """
        self.n_components = n_components
        self.data_reader = data_reader

    @abstractmethod
    def train(self, data=None, labels=None):
        """
        train classifier (fit)

        :return self
        """
        pass

    @abstractmethod
    def test_score(self, test_data=None, test_labels=None):
        """
        train classifier (fit)
        :return tuple of: (score on model for test set, test set)
        """
        pass
