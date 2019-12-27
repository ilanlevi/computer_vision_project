from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from abstract_show_data import AbstractShowData


class AccuracyPlot(AbstractShowData):

    def __init__(self, title='Linear SVM', figure_name=''):
        super(AccuracyPlot, self).__init__(title, figure_name)

    def show_data(self, scores_test_c_list=None):
        """
        Plot the roc plots
        :param scores_test_c_list: list of tuple of (score_res, test_lbl, c_value)
        """
        if scores_test_c_list is None:
            return  # no data, do nothing

        print '#######\nTest Scores for %s \n' % self.title

        for (score_res, test_lbl, c_value) in scores_test_c_list:
            # use sklearn.metrics: roc_curve, roc_auc_score to show roc and auc score
            fpr, tpr, thres = roc_curve(test_lbl, score_res)
            auc = roc_auc_score(test_lbl, score_res)
            #
            name = 'C-Val = %f AUC: %.3f' % (c_value, auc)
            print name
            # name
            plt.plot(fpr, tpr, label=name)

        plt.plot([0, 1], [0, 1], "k--", label='Random Guess')
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate")
        plt.legend(loc="best")
        plt.title("ROC curve - " + self.title)
        plt.show(block=True)
