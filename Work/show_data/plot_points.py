from matplotlib import pyplot as plt

from abstract_show_data import AbstractShowData


class PlotsGraph(AbstractShowData):

    def __init__(self, title='', figure_name=''):
        super(PlotsGraph, self).__init__(title, figure_name)

    def show_data(self, x_s=None, y_s=None, x_label='', y_label=''):
        fig = plt.figure()

        if x_s is None:
            return  # do nothing (no data)
        marker_style = dict(color='tab:blue', linestyle=':', marker='o',
                            markersize=15, markerfacecoloralt='tab:red')
        plt.title(self.title)
        plt.plot(x_s, y_s, **marker_style)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.show(block=False)
