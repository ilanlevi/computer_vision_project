from consts.csv_consts import CsvConsts
from consts.datasets_consts import DataFileConsts as dConsts
from consts.fpn_model_consts import FPNConsts
from data.labeled_data import LabeledData
from models.fpn_wrapper import load_fpn_model, get_3d_pose
from mytools.csv_files_tools import write_csv
from utils.image_tools import save_images

"""
    This is a generating for checking validation set scores
"""


def align_images(cam_m, model_m, dataset):
    scores_vectors = []
    image_list = []

    for i in range(len(dataset.original_file_list)):
        lmarks = dataset.y_train_set[i]
        img = dataset.x_train_set[i]

        splits = dataset.original_file_list[i].split('\\')
        name = splits[-1]

        rx, ry, rz, tx, ty, tz = get_3d_pose(cam_m, model_m, lmarks)

        this_score = [i, name, rx, ry, rz, tx, ty, tz]

        image_list.append((name, img))
        scores_vectors.append(this_score)

    return scores_vectors, image_list


def calc_and_write_scores(folder_path, scores_vec, image_list):
    for csv_row in scores_vec:
        filename = csv_row[1]
        write_csv([csv_row], CsvConsts.CSV_LABELS, folder_path, filename + '.csv')

    # todo add aug
    # todo check error
    # todo check genewration_name
    # todo - add run from console
    save_images(image_list, folder_path)


if __name__ == '__main__':
    # load model
    model_path = FPNConsts.THIS_PATH + FPNConsts.MODELS_DIR
    model_file_name = FPNConsts.POSE_P
    model_name = FPNConsts.MODEL_NAME

    camera_matrix, model_matrix = load_fpn_model(model_path, model_file_name, model_name)

    folder = dConsts.DOWNLOAD_FOLDER
    output_folder = dConsts.OUTPUT_FOLDER
    output_path = folder + output_folder

    output_file = dConsts.OUTPUT_FILE_NAME

    data_sets_list = [dConsts.AFW_DATA, dConsts.IBUG_DATA, dConsts.W300_DATA]

    for data_set in data_sets_list:
        folder_name = data_set['FOLDER']
        suffix = data_set['IMAGE_SUFFIX']
        print 'Generating data for: ' + folder_name

        ds = LabeledData(data_path=folder + folder_name + '\\', picture_suffix=suffix, image_size=500).init()
        ds.read_data_set().filter_multiple_face()
        scores_vector, images_list = align_images(camera_matrix, model_matrix, ds)
        calc_and_write_scores(output_path, scores_vector, images_list)
