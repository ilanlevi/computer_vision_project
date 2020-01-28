from collections import OrderedDict

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

L_EYE = 42  # left eye coordinate
R_EYE = 36  # right eye coordinate

FACIAL_LANDMARKS_68_IDXS_FLIP = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35,
                                 34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41,
                                 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63,
                                 62, 61, 60, 67, 66, 65]
