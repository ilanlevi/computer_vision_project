from Work.tools.my_io import _get_prefix


class FacialLandmarks:

    def __init__(self):
        pass

    @staticmethod
    def get_landmarks(img_path, landmarks_suffix='.pts', print_data=False):
        prefix = _get_prefix(img_path)
        path = prefix + landmarks_suffix
        landmarks = []
        try:
            with open(path, mode='r') as f_marks:
                contents = f_marks.readlines()
                for row in contents:
                    pos = row.rfind(' ')
                    if pos != -1:
                        try:
                            n1 = float(row[:pos])
                            n2 = float(row[pos + 1:])
                            landmarks.append((n1, n2))
                        except Exception as e:
                            if print_data:
                                print 'Nan: %s, Error: %s' % (row, str(e))
        except Exception as e:
            if print_data:
                print 'Error: ' + str(e)

        return landmarks
