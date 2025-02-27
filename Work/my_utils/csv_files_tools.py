import csv
import os
import time


def read_csv(path, filename='', print_data=False):
    """
    Read data from csv
    :param path: path to csv
    :param filename: the file name (with .csv suffix)
    :param print_data: boolean for print time data, default is False
    :return: the data or None if failed
    """
    if print_data:
        print('Starting to read data!')
        print('From: ' + path + filename)

    start = time.time()
    data = []
    line_count = 0

    try:
        path = path + filename

        # with open(path, mode='r') as csv_file:
        with open(path, newline='') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data.append(row)
                line_count += 1

    except Exception as e:
        if print_data:
            print('Read csv failed!')
            print('Reason: ' + str(e))
        return None

    if print_data:
        print('Read CSV completed! (in: %.2f seconds)' % (time.time() - start))
        print('Total lines: %d' % line_count)

    return data


def write_csv(data, fieldnames, path, filename, append=False, print_data=False):
    """
    Write data to csv
    :param fieldnames: csv fields
    :param data: the data
    :param path: path to csv
    :param filename: the file name (with .csv suffix)
    :param append: try append instead of creating new file
    :param print_data: boolean for print time data, default is False
    :return: Succeeded or failed as boolean
    """
    if print_data:
        print('Starting to write data!')
        print('To: ' + path + filename)

    start = time.time()
    line_count = 0
    path = path + filename

    append = append and os.path.exists(path)
    if append:
        mode = 'a'
    else:
        mode = 'w'

    try:
        with open(path, mode=mode, newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if not append:
                # add header
                writer.writeheader()

            for row in data:
                # check labels and data length
                if len(fieldnames) is not len(row):
                    print('Data and fieldnames error! Line: %d' % line_count)
                    return False
                row_to_write = dict()
                for i in range(len(row)):
                    row_to_write[fieldnames[i]] = row[i]

                writer.writerow(row_to_write)
                line_count = line_count + 1

    except Exception as e:
        if print_data:
            print('Write csv failed!')
            print('Reason: ' + str(e))
        return False

    if print_data:
        print('Write CSV completed! (in: %.2f seconds)' % (time.time() - start))
        print('Total lines: %d' % line_count)

    return True
