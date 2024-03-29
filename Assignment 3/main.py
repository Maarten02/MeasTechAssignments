import numpy as np
from tabulate import tabulate, SEPARATING_LINE
import matplotlib.pyplot as plt
import math
from cross_correlate import find_pixel_shift

from tifffile import imread

def print_lists(names, vals):
    ptb = []
    for name, val in zip(names, vals):
        line = [name, '|'] + list(val)
        ptb.append(line)
    print(tabulate(ptb))


pt = [['Problem', 'Parameter', 'Value', 'Unit']]

execute_problem1 = True
execute_problem2 = True
execute_problem3 = True

# ======================================================================
# =                           Problem 1                                =
# ======================================================================
if execute_problem1:

    # ================= 1 (a) ====================== []
    I_A = np.array([50, 55, 72, 50, 40, 33])
    I_B = np.array([51, 50, 56, 75, 54, 43])

    I_A_fluc = I_A - np.average(I_A)
    I_B_fluc = I_B - np.average(I_B)

    print_lists(['I_A_fluc', 'I_B_fluc'], [I_A_fluc, I_B_fluc])

    # ================= 1 (b) ====================== []
    pixel_shift_tbl = np.arange(-5, 6)

    # ================= 1 (c) ====================== []
    # fluctuations
    R = np.empty(len(pixel_shift_tbl))
    for i, pix_shift in enumerate(pixel_shift_tbl):
        corsum = 0
        for j, pix_A in enumerate(I_A_fluc):
            pix_B_idx = j + pix_shift
            if 0 < pix_B_idx < 6:
                corsum += pix_A * I_B_fluc[pix_B_idx]

        R[i] = 1 / 6 * corsum  # - abs(pix_shift)

    print_lists(['pixel shift table (fluc)', 'R (fluc)'], [pixel_shift_tbl, R])

    # absolute --> magnitude of correlations much greater...
    # since sign of fluctuation is not accounted for
    # R = np.empty(len(pixel_shift_tbl))
    # for i, pix_shift in enumerate(pixel_shift_tbl):
    #     corsum = 0
    #     for j, pix_A in enumerate(I_A):
    #         pix_B_idx = j + pix_shift
    #         if 0 < pix_B_idx < 6:
    #             corsum += pix_A * I_B[pix_B_idx]
    #
    #     R[i] = 1 / 6 * corsum  #  - abs(pix_shift)
    #
    # print_lists(['pixel shift table', 'R'], [pixel_shift_tbl, R])

    # ================= 1 (d) ====================== []
    R_1_idx = np.argmax(R)
    R_2_idx = R_1_idx - 1
    R_3_idx = R_1_idx + 1

    x_p = (pixel_shift_tbl[R_1_idx] + 1 / 2 * ((np.log(R[R_2_idx]) - np.log(R[R_3_idx]))
                                               / (np.log(R[R_2_idx])
                                                  + np.log(R[R_3_idx])
                                                  - 2 * np.log(R[R_1_idx]))))

    pt.append(['1d', 'R1_idx', R_1_idx, '-'])
    pt.append(['1d', 'x_p', x_p, '-'])

    # ================= 1 (e) ====================== []
    distance = x_p * 0.5e-3
    time = 0.01
    velocity = distance / time

    pt.append(['1e', 'velocity', velocity, 'm/s'])

# ======================================================================
# =                           Problem 2                                =
# ======================================================================
if execute_problem2:

    # ================= 2 (a) ====================== []
    sw_a = imread('data/Single_window_a.tiff')
    sw_b = imread('data/Single_window_b.tiff')

    delta_y, delta_x = find_pixel_shift(sw_a, sw_b)


print(tabulate(pt))
