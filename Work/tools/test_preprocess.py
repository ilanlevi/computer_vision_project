from random import randint
import unittest
# imports
from Work.consts.files_consts import FileConsts
from Work.consts.ds_consts import DataSetConsts
from Work.data.helen_data import HelenDataSet
from external_face_pos import PreProcessDataExternal
from image_tools import ImageTools


class MyTestCase(unittest.TestCase):

    def test_load_data(self):
        """
        Test on one picture
        """
        ds = HelenDataSet(data_path=FileConsts.DOWNLOAD_FOLDER, original_sub=FileConsts.DOWNLOAD_SUB_FOLDER,
                          target_sub=FileConsts.PROCESSED_SET_FOLDER)
        ds.init()
        original_images = ds.original_file_list

        # test size
        self.assertNotEqual(len(original_images), 0)
        return ds

    def test_show_feature_on_image(self):
        ds = HelenDataSet(data_path=FileConsts.DOWNLOAD_FOLDER, original_sub=FileConsts.DOWNLOAD_SUB_FOLDER,
                          target_sub=FileConsts.PROCESSED_SET_FOLDER)
        ds.init()
        original_images = ds.original_file_list

        rnd_index = randint(0, len(original_images) - 1)
        images = ImageTools.load_images([ds.original_file_list[rnd_index]], DataSetConsts.PICTURE_WIDTH)
        con_images = ImageTools.load_converted_images([ds.original_file_list[rnd_index]],
                                                      DataSetConsts.PICTURE_WIDTH)

        pr = PreProcessDataExternal(predictor_path=(FileConsts.DOWNLOAD_FOLDER + FileConsts.PREDICTOR_FILE_NAME))

        rect = pr.get_shapes(con_images[0])
        print ('Rectangles for face:\n%s' % str(rect))

        bb = PreProcessDataExternal.draw_on_image(pr.predictor, pr.detector, images[0], con_images[0])

        print str(bb)

        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
