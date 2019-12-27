import matplotlib.pyplot as plt
import numpy as np
from tmp.classification.pca import MyPCA
from tmp.data.mnist_data import MNIST
from tmp.show_data.images_as_sub_plots import ImagesGraph
from tmp.knn_tools import KNNTools
from Work.tools.pca_tools import PCATools

PATH_TO_MNIST = "C:\\Users\\ilan_\\Downloads\\Browsers\\section2data\\"
MNIST_NAME = 'mnist.gz'

TRAIN_SIZE = 12000
TEST_SIZE = 2000
SELECTED_IMAGE_INDEX_PART_G = 123

RUN_PART_B = False
RUN_PART_C = False
RUN_PART_D = False
RUN_PART_E = False
RUN_PART_F = False
RUN_PART_G = True
RUN_PART_H = False

if __name__ == '__main__':
    # TEST PCA
    md = MNIST(data_path=PATH_TO_MNIST, file_name=MNIST_NAME).init()
    md.set_size(train_size=TRAIN_SIZE, test_size=TEST_SIZE)
    pca_tools = PCATools(md)

    my_pca = None

    # run part b
    if RUN_PART_B:
        n_components = 6
        # create pca
        my_pca = MyPCA(data_reader=md)
        my_pca.train()
        # show 6 first principal components
        pca_tools.plot_n_principal_components(n_components, my_pca.pca)
        # show mean image
        pca_tools.show_mean_image(my_pca.pca)

    # run part c
    if RUN_PART_C:
        # create pca if None
        if my_pca is None:
            my_pca = MyPCA(data_reader=md)
            my_pca.train()
        # show percentage variance explained graph
        pca_tools.show_percentage_var_explained(my_pca.pca)

    # run part d
    if RUN_PART_D:
        variances = [.95, .80]
        data = md.x_train_set
        results = PCATools.number_of_required_components_for_variance(data, variances)
        # display results:
        print '\n####Question 2 section d:\n'
        for (var, n_comp) in results:
            print ('For %.2f variance: %d components are required' % (var, n_comp))

    if RUN_PART_E:
        n_components = 2
        # create pca if None
        if my_pca is None:
            my_pca = MyPCA(data_reader=md)
            my_pca.train()
        # project the data into 2-nd dimension
        c_list = my_pca.projected_data(n_components, md.x_train_set)
        # plot the result
        PCATools.plot_2_dimensions(c_list, md.y_train_set)

    if RUN_PART_F:
        # set up dimensions
        dimensions = [2, 10, 20]
        # knn settings
        k_max = 11
        tmp_md = MNIST(data_path=PATH_TO_MNIST, file_name=MNIST_NAME).init()
        tmp_md.set_size(train_size=TRAIN_SIZE, test_size=TEST_SIZE)
        # TEST KNN
        pca_tools.run_knn_on_projected_data(dimensions, k_max, tmp_md)

    if RUN_PART_G:
        selected_index = SELECTED_IMAGE_INDEX_PART_G

        # set up dimensions
        dimensions = [2, 5, 10, 50, 100, 150]

        pca_tools.project_digit_to_dimensions(dimensions, md, selected_index)

    if RUN_PART_H:
        # part 1
        n_components = 6  # number of components
        selected_index = 123  # selected image index
        original_image = md.x_test_set[selected_index]
        # create all digit pca
        all_digit_pca = MyPCA(n_components, md).train()
        all_digit_approx, _ = all_digit_pca.approximate_data(n_components, md.x_test_set)
        all_digit_approx = all_digit_approx[selected_index]
        all_digit_approx = np.asarray(all_digit_approx)
        # create pca for each digit
        pca_arr = pca_tools.create_pca_per_digit(md, n_components)

        transformed = []
        approximations = []

        for i in range(10):
            # part 1 - plot 6 first pc for each
            pca_tools.plot_n_principal_components(n_components=n_components, pca=pca_arr[i].pca,
                                                  title=('PCA for digit = %d' % i), show_plt=False)
            # part 2 and 3
            # transform and invert (reconstruction) all images to 6 components
            a, t = pca_arr[i].approximate_data(n_dimensions=n_components, x_data=md.x_test_set, fit=False)
            t = np.asarray(t)
            a = np.asarray(a)
            # save results
            transformed.append(t)
            approximations.append(a)

        # part 1 - mean of differences between of all digit model to specific digit
        print '####\nMean of differences between digits components models to all digit model:'
        # print mean between specific digit components models to all digit model
        for i in range(10):
            print 'Digit [%d] model mean = %.3f' % \
                  (i, np.mean(np.mean(
                      all_digit_pca.pca.components_[:n_components] - pca_arr[i].pca.components_[:n_components])))

        # get reconstructed image from all of the PCA's
        reconstructed_images = [approximations[i][selected_index] for i in range(10)]
        # add original
        reconstructed_images.append(original_image)

        # show reconstructed images
        sub_labels = ['From %d' % i for i in range(10)]
        sub_labels.append('Original')
        reconstructed_images = np.asarray(reconstructed_images)
        s = md.get_picture_size()
        reconstructed_images_reshaped = np.reshape(reconstructed_images, (11, s, s))
        g = ImagesGraph(title='PCA inverse on image from different models', col_size=3)
        g.show_data(images=reconstructed_images_reshaped, images_labels=sub_labels, show_plt=False)

        # calculate mean between the original's to the reconstructed from models
        diff_arr = [np.mean(np.mean((md.x_test_set - approximations[i]))) for i in range(10)]
        # add all digits
        diff_arr.append(np.mean(np.mean((md.x_test_set - all_digit_approx))))

        print '####\nMean differences between original images to models:'
        print 'All digit model mean = %.3f' % diff_arr[10]
        for i in range(10):
            print 'digit [%d] model mean = %.3f' % (i, diff_arr[i])

        i = int(np.argmin(diff_arr))
        if i is 10:
            # if all digit model won
            print("!@#!@#!@\nAll digit model achieved lowest mean of %.3f" % diff_arr[0])
            # no need to run the same knn again
        else:
            # another digit won
            print("!@#!@#!@\nDigit %d model achieved lowest mean of %.3f " % (i, diff_arr[i]))
            # create db
            tmp_md = MNIST(data_path=PATH_TO_MNIST, file_name=MNIST_NAME).init()
            tmp_md.set_size(train_size=TRAIN_SIZE, test_size=TEST_SIZE)
            # transform db to fit selected model
            tmp_md.x_train_set, _ = pca_arr[i].approximate_data(n_dimensions=n_components, x_data=md.x_train_set,
                                                                fit=False)
            tmp_md.x_test_set, _ = pca_arr[i].approximate_data(n_dimensions=n_components, x_data=md.x_test_set,
                                                               fit=False)
            # test knn model on 10 different k values
            knn = KNNTools(k_max=11, mnist=tmp_md)
            knn.test_knn_range()

    plt.show()
