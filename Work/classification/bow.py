import time

import numpy as np
from sklearn.preprocessing import StandardScaler


class Bow:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.descriptor_stack = None
        self.vocabulary = None
        self.scale = StandardScaler()

    def generate_vocabulary(self, n_images, descriptor_list, k_means_returned):
        """
            Generate a vocabulary - Every image represented as a combination of multiple visual words.
            (Sets self.vocabulary)
            :param n_images: images
            :param descriptor_list: descriptor (d-sift)
            :param k_means_returned: k-means output - index's of the cluster each sample belongs to.
            :return self
        """
        start = time.time()
        self.vocabulary = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
        old_count = 0
        for i in range(n_images):
            le = len(descriptor_list[i])
            for j in range(le):
                idx = k_means_returned[old_count + j]
                self.vocabulary[i][idx] += 1
            old_count += le
        print ('Bow creation - took: %.2f sec.' % (time.time() - start))
        return self

    def standardize(self):
        """
            Normalize the vocabulary for better classifier performance
            :return self
        """
        start = time.time()

        self.scale = StandardScaler().fit(self.vocabulary)
        self.vocabulary = self.scale.transform(self.vocabulary)

        print ('Bow standardize - took: %.2f sec.' % (time.time() - start))

    def format_nd(self, l):
        """
            restructures list into v-stack array of shape (M x N)
            M  - samples
            N - features
        """
        v_stack = np.array(l[0])
        for remaining in l[1:]:
            v_stack = np.vstack((v_stack, remaining))
        self.descriptor_stack = v_stack.copy()
        return v_stack
