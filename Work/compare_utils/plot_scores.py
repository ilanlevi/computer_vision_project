import matplotlib.pyplot as plt
import numpy as np

from consts.csv_consts import CsvConsts
from mytools.csv_files_tools import read_csv


def plot_diff_each_param(folder, file_list):
    """
    Plot csv scores, each param of 6DoF in a different axes
    :param folder: the directory of the files
    :param file_list: file names list to compare
    :return: Nan - show data
    """
    fields = CsvConsts.CSV_VALUES_LABELS
    csv_values = []
    csv_lengths = []
    for f_name in file_list:
        csv_file = read_csv(folder, f_name)
        csv_value = [[float(fileRow[field]) for fileRow in csv_file] for field in fields]
        csv_values.append(csv_value)
        csv_lengths.append(len(csv_file))

    csv_lengths = np.asarray(csv_lengths)

    x_s = range(csv_lengths.min())
    fig, axs = plt.subplots(len(fields), 1)

    for i in range(len(fields)):
        for csv_index in range(len(csv_values)):
            axs[i].plot(x_s, csv_values[csv_index][i], label=(fields[i] + '  ' + file_list[csv_index]), alpha=0.5)

        axs[i].grid(True)
        axs[i].legend()


def plot_diff(folder, filename, title=''):
    """
    Plot the difference file scores
    :param folder: the directory of the files
    :param filename: the difference file name to compare
    :param title: figure title
    :return: None - plot the results
    """
    diff = read_csv(folder, filename)
    x_s = range(len(diff))

    R_X = [float(i[CsvConsts.R_X]) for i in diff]
    R_Y = [float(i[CsvConsts.R_Y]) for i in diff]
    R_Z = [float(i[CsvConsts.R_Z]) for i in diff]
    T_X = [float(i[CsvConsts.T_X]) for i in diff]
    T_Y = [float(i[CsvConsts.T_Y]) for i in diff]
    T_Z = [float(i[CsvConsts.T_Z]) for i in diff]

    has_theta = diff[0].has_key(CsvConsts.THETA)
    if has_theta:
        THETA = [float(i[CsvConsts.THETA]) for i in diff]
    else:
        THETA = [float(0) for i in diff]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.suptitle(title)

    # show rotation difference
    ax1.plot(x_s, R_X, label=CsvConsts.R_X, alpha=0.5)
    ax1.plot(x_s, R_Y, label=CsvConsts.R_Y, alpha=0.5)
    ax1.plot(x_s, R_Z, label=CsvConsts.R_Z, alpha=0.5)
    ax1.grid(True)
    ax1.legend()

    # show translation difference
    ax2.plot(x_s, T_X, label=CsvConsts.T_X, alpha=0.5)
    ax2.plot(x_s, T_Y, label=CsvConsts.T_Y, alpha=0.5)
    ax2.plot(x_s, T_Z, label=CsvConsts.T_Z, alpha=0.5)
    ax2.grid(True)
    ax2.legend()

    # show theta result
    ax3.plot(x_s, THETA, label=CsvConsts.THETA, alpha=0.5)
    ax3.grid(True)
    ax3.legend()
