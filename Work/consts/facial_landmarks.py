from collections import OrderedDict

# TODO - DELTE

class FacialLandmarksConsts:
    FACIAL_LANDMARKS_68_INDEXES = OrderedDict([
        ("mouth", (48, 68)),
        ("right_eyebrow", (17, 22)),
        ("left_eyebrow", (22, 27)),
        ("right_eye", (36, 42)),
        ("left_eye", (42, 48)),
        ("nose", (27, 36)),
        ("jaw", (0, 17))
    ])

    FACIAL_LANDMARKS_5_INDEXES = OrderedDict([
        ("right_eye", (2, 3)),
        ("left_eye", (0, 1)),
        ("nose", 4)
    ])

    def __init__(self):
        pass
