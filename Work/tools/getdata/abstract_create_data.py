from abc import ABCMeta, abstractmethod
from ..csv_files_tools import write_csv
from Work.consts import CsvConsts


class AbstractConvertData:
    """
    This class will be used as an interface to convert labels to my label format
    """
    __metaclass__ = ABCMeta

    def __init__(self, data_path='', output_file='', print_data=False):
        """
        init the object
        :param data_path: the full path to files
        :param output_file: the file to write (with full path)
        :param print_data: print data or not, default is false
        """
        self.data_path = data_path
        self.output_path = output_file
        self.print_data = print_data

    def write_my_format(self, labels_formatted):
        """
        write
        :param labels_formatted: the data in my format (see csv_consts.py)
        :return: succeeded or not
        """
        success = write_csv(labels_formatted, CsvConsts.CSV_LABELS, self.output_path, '', print_data=self.print_data)
        return success

    @abstractmethod
    def convert_format(self):
        """
        convert the images labels to my label format
        :return:
        """
        pass
