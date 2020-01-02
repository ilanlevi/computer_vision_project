from random import randint
import unittest
import cv2
import numpy as np

from ..consts.files_consts import HelenFileConsts as hfc
from ..consts.ds_consts import DataSetConsts
from ..data.helen_data import HelenDataSet
from ..utils.my_io import _load
from external_face_pos import PreProcessDataExternal
from image_tools import ImageTools
from ..utils.images_pose import ImagesPose as ip


class MyTestCase(unittest.TestCase):

    @staticmethod
    def generate_dataset():
        # ds = HelenDataSet(data_path=hfc.DOWNLOAD_FOLDER, original_sub=hfc.DOWNLOAD_SUB_FOLDER,
        #                   target_sub=hfc.PROCESSED_SET_FOLDER)
        ds = HelenDataSet(data_path=hfc.DOWNLOAD_FOLDER2, original_sub=hfc.VALID_SET_SUB_FOLDER,
                          target_sub=hfc.PROCESSED_SET_FOLDER, picture_suffix='png')
        ds.init()
        return ds

    def test_load_data(self):
        """
        Test on one picture
        """
        ds = MyTestCase.generate_dataset()
        original_images = ds.original_file_list

        # test size
        self.assertNotEqual(len(original_images), 0)
        return ds

    def test_show_feature_on_image(self):
        ds = MyTestCase.generate_dataset()
        original_images = ds.original_file_list

        rnd_index = randint(0, len(original_images) - 1)
        images = ImageTools.load_images([ds.original_file_list[rnd_index]], DataSetConsts.PICTURE_WIDTH)
        con_images = ImageTools.load_converted_images([ds.original_file_list[rnd_index]],
                                                      DataSetConsts.PICTURE_WIDTH)

        pr = PreProcessDataExternal(
            predictor_path=(hfc.DOWNLOAD_FOLDER + hfc.PREDICTOR_FILE_NAME))

        rect = pr.get_shapes(con_images[0])
        print ('Rectangles for face:\n%s' % str(rect))

        bb = PreProcessDataExternal.draw_on_image(pr.predictor, pr.detector, images[0], con_images[0])

        print str(bb)

        self.assertEqual(True, True)

    NUMBER_OF_TESTS = 1

    def test_align_image(self):
        ds = MyTestCase.generate_dataset()
        original_images = ds.original_file_list
        pr = PreProcessDataExternal(
            predictor_path=(hfc.DOWNLOAD_FOLDER + hfc.PREDICTOR_FILE_NAME))

        for i in range(MyTestCase.NUMBER_OF_TESTS):
            rnd_index = randint(0, len(original_images) - 1)
            image_original = ImageTools.load_images([ds.original_file_list[rnd_index]], width=None)
            images = ImageTools.load_images([ds.original_file_list[rnd_index]], DataSetConsts.PICTURE_WIDTH)
            con_images = ImageTools.load_converted_images([ds.original_file_list[rnd_index]],
                                                          DataSetConsts.PICTURE_WIDTH)

            cv2.imshow(("The Original #%d" % i), image_original[0])
            rects = pr.detector(con_images[0], 2)

            for rect in rects:
                # extract the ROI of the *original* face, then align the face
                # using facial landmarks
                (x, y, w, h) = pr.rect_to_bb(rect)
                face_orig = ImageTools.resize(images[0][y:y + h, x:x + w], width=256)
                face_aligned = pr.align(images[0], con_images[0], rect)

                print 'Image rect: #%d: %s' % (i, str(rect))

                # display the output images
                cv2.imshow("Original #%d %s" % (i, str(rect)), face_orig)
                cv2.imshow("Aligned #%d %s" % (i, str(rect)), face_aligned)

        cv2.waitKey(0)
        self.assertEqual(True, True)

    # def test_face_pose(self):
    #     ds = MyTestCase.generate_dataset()
    #     original_images = ds.original_file_list
    #     pr = PreProcessDataExternal(
    #         predictor_path=(hfc.DOWNLOAD_FOLDER + hfc.PREDICTOR_FILE_NAME))
    #
    #     meta = _load(hfc.DOWNLOAD_FOLDER + hfc.SETTING_PKL_FILE_NAME)
    #
    #     for i in range(MyTestCase.NUMBER_OF_TESTS):
    #         rnd_index = randint(0, len(original_images) - 1)
    #         image_original = ImageTools.load_images([ds.original_file_list[rnd_index]], width=None)
    #         images = ImageTools.load_images([ds.original_file_list[rnd_index]], DataSetConsts.PICTURE_WIDTH)
    #         con_images = ImageTools.load_converted_images([ds.original_file_list[rnd_index]],
    #                                                       DataSetConsts.PICTURE_WIDTH)
    #
    #         cv2.imshow(("The Original #%d - %s" % (i, original_images[rnd_index])), image_original[0])
    #         rects = pr.detector(con_images[0], 1)
    #         for rect in rects:
    #             # extract the ROI of the *original* face, then align the face
    #             # using facial landmarks
    #             pts = pr.predictor(image_original, rect).parts()
    #             pts = np.array([[pt.x, pt.y] for pt in pts]).T
    #             roi_box = ip.parse_roi_box_from_landmark(pts)
    #
    #             img = ImageTools.crop_img(image_original[0], roi_box)
    #
    #             img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
    #
    #             # input = transform(img).unsqueeze(0)
    #             pts68 = predict_68pts(param, roi_box)
    #
    #             face_orig = ImageTools.resize(images[0][y:y + h, x:x + w], width=256)
    #             face_aligned, roi = pr.align_get_matrix(images[0], con_images[0], rect)
    #             p, pose = ip.parse_pose(roi, meta)
    #             print 'Image rect: #%d: %s' % (i, str(rect))
    #
    #             # display the output images
    #             cv2.imshow("Original #%d %s" % (i, str(rect)), face_orig)
    #             cv2.imshow("Aligned #%d %s" % (i, str(rect)), face_aligned)
    #             # print "Aligned #%d %s" % (i, str(rect)), face_aligned
    #
    #             print 'P = ' + str(p)
    #             print '6Dof = ' + str(pose)
    #
    #     cv2.waitKey(0)
    #     self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
