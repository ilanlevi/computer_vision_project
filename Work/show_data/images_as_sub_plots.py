import math

from matplotlib import pyplot as plt

from abstract_show_data import AbstractShowData


class ImagesGraph(AbstractShowData):

    def __init__(self, title=None, col_size=1):
        super(ImagesGraph, self).__init__(title)
        self.col_size = col_size

    # abstract

    def show_data(self, images=None, images_labels=None, max_images=None, show_data_points=False,
                  show_plt=True):

        if images_labels is None:
            return  # do nothing (no data)

        if max_images is None:
            max_images = len(images)

        rows = 1 + int(math.ceil((max_images + 0.0) / self.col_size))

        fig, axs = plt.subplots(nrows=rows, ncols=self.col_size)
        # disable axis
        for ax_row in axs:
            for ax in ax_row:
                ax.axis('off')
        # set title
        if self.title is None:
            # if no title, start from first row
            starting_row = 0
        else:
            axs[0][self.col_size / 2].set_title(self.title)
            starting_row = 1

        for index in range(max_images):
            if index < len(images):
                row = starting_row + (index / self.col_size)
                col = index % self.col_size
                axs[row][col].imshow(images[index], cmap='gray', interpolation='nearest')
                if images_labels is not None:
                    axs[row][col].set_title(images_labels[index])

        plt.show(block=show_plt)
