import numpy as np
from tabulate import tabulate, SEPARATING_LINE
import matplotlib.pyplot as plt
import math
from cross_correlate import find_pixel_shift
from binning import find_velocity_field
from plotting import Plotting
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
    sw_a = np.flipud(sw_a)
    sw_b = np.flipud(sw_b)

    delta_y, delta_x = find_pixel_shift(sw_a, sw_b, plot=True)

    pt.append(['2a', 'delta x', delta_x, 'pixels'])
    pt.append(['2a', 'delta y', delta_y, 'pixels'])

    # ================= 2 (b) ====================== []
    im_a = imread('data/LOvort_0001_a.tif')
    im_b = imread('data/LOvort_0001_b.tif')

    # u_arr, v_arr, x_pos_vel, y_pos_vel = find_velocity_field(im_a, im_b, binsize, overlap)
    # X, Y = np.meshgrid(x_pos_vel, y_pos_vel)
    # plt.quiver(X, Y, u_arr, v_arr)
    # plt.show()

    # ================= 2 (c) ====================== []
    # since the displacement is in the order of 1 px, the gaussian is necessary to
    # provide the necessary precision, otherwise the velocity is quantized very coarsely

    # ================= 2 (d) ====================== []
    # the flow is a vortex

    # ================= 2 (e) ====================== []
    # plot_2e = Plotting(im_a, im_b, 32, 0.5)
    # plot_2e.save()

    # ================= 2 (f) ====================== []
    # shows difference in grid, same spacing but slightly shifted
    # Not very clear what actual difference is

    # ================= 2 (g) ====================== []
    # This one has way too low resolution
    # plot_2g_64_00 = Plotting(im_a, im_b, 64, 0)
    # plot_2g_64_00.save('vortex')

    # This one messes up the vorticity, velocity seems OK
    # plot_2g_16_75 = Plotting(im_a, im_b, 16, 0.75)
    # plot_2g_16_75.save('vortex')

    # ================= 2 (h) ====================== []
    im_cyl_a = imread('data/Cylinder_a.tif')
    im_cyl_b = imread('data/Cylinder_b.tif')
    plot_2h = Plotting(im_cyl_a, im_cyl_b, 32, 0.5)
    plot_2h.save('cylinder')

    # if you don't take the fluctuations correlation results in spurious results
print(tabulate(pt))
