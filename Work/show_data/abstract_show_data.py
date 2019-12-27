from abc import ABCMeta, abstractmethod


class AbstractShowData:
    __metaclass__ = ABCMeta

    def __init__(self, title='', figure_name=''):
        self.title = title
        self.figure_name = figure_name

    @abstractmethod
    def show_data(self):
        """
            plot the data
        """
        pass
