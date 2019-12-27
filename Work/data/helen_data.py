import os

from abstract_read_data import AbstractReadData

from Work.consts.ds_consts import DataSetConsts


class HelenDataSet(AbstractReadData):

    def __init__(self, data_path, original_sub, target_sub, random_state=DataSetConsts.DEFAULT_RANDOM_STATE,
                 train_rate=DataSetConsts.DEFAULT_TRAIN_RATE, image_size=DataSetConsts.PICTURE_WIDTH,
                 picture_suffix=DataSetConsts.PICTURE_SUFFIX):
        super(HelenDataSet, self).__init__(data_path, random_state, train_rate, image_size)
        self.data_path = data_path
        self.original_sub = original_sub
        self.target_sub = target_sub
        self.random_state = random_state
        self.train_rate = train_rate
        self.image_size = image_size
        self.original_file_list = []
        self.target_file_list = []
        self.picture_suffix = picture_suffix

    def get_original_list(self):
        """
            return the dataset list of images from self.data_path +  self.original_sub
            :return self
        """

        path = self.data_path + self.original_sub
        files = [os.path.join(r, file_) for r, d, f in os.walk(path) for file_ in f]
        f_list = []
        for file_name in files:
            if self.picture_suffix in file_name:
                f_list.append(file_name)
        return f_list

    @staticmethod
    def get_file_type(file_name):
        """
            Split file label from the name (data set structure)
            :return the file label
        """
        split, _ = file_name.split('_', 1)
        return split

    #
    # def load_types(self, type1=None, type2=None, path_to_use=None):
    #     """
    #         Load the dataset image names for 2 types.
    #         If type1 or type2 is None, use self values.
    #         :return dict('type1': list_of_type1_file_names, 'type2': list_of_type2_file_names)
    #     """
    #     if type1 is not None:
    #         self.type1_name = type1
    #     if type2 is not None:
    #         self.type2_name = type2
    #     if path_to_use is None:
    #         path_to_use = self.full_path
    #
    #     different_types = {self.type1_name: [], self.type2_name: []}
    #     local_path = '%s%s\\' % (path_to_use, self.data_path)
    #     try:
    #         # read all files in dir
    #         files = os.listdir(local_path)
    #         # remove special file from list
    #         files.remove('Thumbs.db')
    #         for file_name in files:
    #             # check image type (by the prefix value)
    #             file_type = Spatial.get_file_type(file_name)
    #             if file_type in different_types:
    #                 # if the file is the required type, mark to load it
    #                 different_types.get(file_type).append(file_name)
    #
    #     except Exception as e:
    #         print 'Error while reading files! \n##Path = %s\nError: %s' % (local_path, str(e))
    #
    #     return different_types

    # abstracts:

    def read_data_set(self, unzip_file=True, path_to_use=None):
        # if unzip_file:
        #     self.unzip_file()
        # return self.load_types(path_to_use=path_to_use)
        return self

    def read_data_set_and_separate(self, unzip_file=True, path_to_use=None):
        # if path_to_use is None:
        #     path_to_use = self.full_path
        # x_set = []
        # y_set = []
        #
        # # get files map to read
        # mp = self.read_data_set(unzip_file=unzip_file, path_to_use=path_to_use)
        # local_path = '%s%s\\' % (path_to_use, self.data_path)
        # # read all files
        # for k in mp.keys():
        #     for f in mp.get(k):
        #         try:
        #             image_path = local_path + f
        #             im = cv2.imread(image_path)
        #             im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        #             x_set.append(im)
        #             y_set.append(k)
        #         except Exception as e:
        #             print 'Error while reading file %s! %s' % (local_path, str(e))
        #
        # self.x_train_set = np.asarray(x_set)
        # self.y_train_set = np.asarray(y_set)

        return self

    def _init(self, unzip_file=True, path_to_use=None):
        self.original_file_list = self.get_original_list()
        return self


if __name__ == '__main__':
    """
    Test loading images 
    """
    from Work.consts.files_consts import FileConsts

    ds = HelenDataSet(data_path=FileConsts.DOWNLOAD_FOLDER, original_sub=FileConsts.DOWNLOAD_SUB_FOLDER,
                      target_sub=FileConsts.PROCESSED_SET_FOLDER)
    ds.init()
    # Print number of images
    print str(len(ds.original_file_list))

