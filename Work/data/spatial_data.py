import os
import zipfile

import cv2
import numpy as np

from abstract_read_data import AbstractReadData


class Spatial(AbstractReadData):
    # const
    DATA_SET_PATH_PART_2 = 'part2_dataset.zip'
    DATA_SET_EXTRACT_TARGET = 'section2data'
    DATA_SET_SUB_FOLDER = 'spatial_envelope_256x256_static_8outdoorcategories'
    DEFAULT_TRAIN_RATE = 1 / 4.0

    PICTURE_SIZE = 256

    def __init__(self, full_path='', data_path=DATA_SET_EXTRACT_TARGET + "\\" + DATA_SET_SUB_FOLDER,
                 target_path=DATA_SET_EXTRACT_TARGET, archive_path=DATA_SET_PATH_PART_2, type1_name='coast',
                 type2_name='forest', train_rate=DEFAULT_TRAIN_RATE):
        super(Spatial, self).__init__(full_path)
        self.full_path = full_path
        self.image_size = Spatial.PICTURE_SIZE
        self.data_path = data_path
        self.archive_path = archive_path
        self.target_path = target_path
        self.train_rate = train_rate
        self.type1_name = type1_name
        self.type2_name = type2_name

    def unzip_file(self):
        """
            Unzip's the dataset zipped file into self.target_path path
            :return self
        """
        try:
            p = self.full_path + self.archive_path
            with zipfile.ZipFile(p, 'r') as zip_ref:
                zip_ref.extractall(self.full_path + self.target_path)
                zip_ref.close()
        except Exception as e:
            print 'Error while extracting file! ' + str(e)
        return self

    @staticmethod
    def get_file_type(file_name):
        """
            Split file label from the name (data set structure)
            :return the file label
        """
        split, _ = file_name.split('_', 1)
        return split

    def load_types(self, type1=None, type2=None, path_to_use=None):
        """
            Load the dataset image names for 2 types.
            If type1 or type2 is None, use self values.
            :return dict('type1': list_of_type1_file_names, 'type2': list_of_type2_file_names)
        """
        if type1 is not None:
            self.type1_name = type1
        if type2 is not None:
            self.type2_name = type2
        if path_to_use is None:
            path_to_use = self.full_path

        different_types = {self.type1_name: [], self.type2_name: []}
        local_path = '%s%s\\' % (path_to_use, self.data_path)
        try:
            # read all files in dir
            files = os.listdir(local_path)
            # remove special file from list
            files.remove('Thumbs.db')
            for file_name in files:
                # check image type (by the prefix value)
                file_type = Spatial.get_file_type(file_name)
                if file_type in different_types:
                    # if the file is the required type, mark to load it
                    different_types.get(file_type).append(file_name)

        except Exception as e:
            print 'Error while reading files! \n##Path = %s\nError: %s' % (local_path, str(e))

        return different_types

    # abstracts:

    def read_data_set(self, unzip_file=True, path_to_use=None):
        if unzip_file:
            self.unzip_file()
        return self.load_types(path_to_use=path_to_use)

    def read_data_set_and_separate(self, unzip_file=True, path_to_use=None):
        if path_to_use is None:
            path_to_use = self.full_path
        x_set = []
        y_set = []

        # get files map to read
        mp = self.read_data_set(unzip_file=unzip_file, path_to_use=path_to_use)
        local_path = '%s%s\\' % (path_to_use, self.data_path)
        # read all files
        for k in mp.keys():
            for f in mp.get(k):
                try:
                    image_path = local_path + f
                    im = cv2.imread(image_path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    x_set.append(im)
                    y_set.append(k)
                except Exception as e:
                    print 'Error while reading file %s! %s' % (local_path, str(e))

        self.x_train_set = np.asarray(x_set)
        self.y_train_set = np.asarray(y_set)

        return self

    def _init(self, unzip_file=True, path_to_use=None):
        return self.read_data_set_and_separate(unzip_file=unzip_file, path_to_use=path_to_use).split_dataset()
