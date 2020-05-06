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

L_EYE_IDX = 45  # left eye coordinate
R_EYE_IDX = 36  # right eye coordinate

# Index to remap landmarks in case we flip an image
FACIAL_LANDMARKS_68_IDXS_FLIP = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25, \
                                 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43, \
                                 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56, \
                                 65, 64, 63, 62, 61, 68, 67, 66]
