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

L_EYE = 45  # left eye coordinate
R_EYE = 36  # right eye coordinate

FACIAL_LANDMARKS_68_IDXS_FLIP = [(16, 0), (15, 1), (14, 2), (13, 3), (12, 4), (11, 5), (10, 6), (9, 7),  # jaw
                                 (26, 17), (25, 18), (24, 19), (23, 20), (22, 21),  # eyebrows
                                 (35, 31), (34, 32),  # nose
                                 (45, 36), (44, 37), (43, 38), (42, 39), (46, 41), (47, 40),  # eyes
                                 # outer mouth
                                 (48, 54), (49, 53), (50, 52),  # upper mouth
                                 (59, 55), (58, 56),  # lower mouth
                                 # inner mouth
                                 (60, 64), (61, 63),  # upper mouth
                                 (67, 65)  # lower mouth
                                 ]
