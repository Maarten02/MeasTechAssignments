import numpy as np
import matplotlib.pyplot as plt


def find_pixel_shift(bin_a, bin_b, plot=False):

    """
    This function takes:
     bin_a: containing the pixels from current timestep t
     bin_b: containing the pixels from next timestep t + Delta t
     plot: option to plot heatmap of correlation

    And returns:
     shift_i: the shift in the y direction (downwards positive) from timestep t to timestep t + Delta t
     shift_j: the shift in the x direction (right positive) from timestep t to timestep t + Delta t
    """

    fluc_a = bin_a - np.mean(bin_a)
    fluc_b = bin_b - np.mean(bin_b)

    M, N = sw_a.shape
    ov = 0.75

    max_x_shift = int((1 - ov) * N)
    max_y_shift = int((1 - ov) * M)

    x_shift_arr = np.tile(np.arange(-1 * max_x_shift, max_x_shift + 1), (2 * max_y_shift + 1, 1))
    y_shift_arr = np.tile(np.arange(-1 * max_y_shift, max_y_shift + 1), (2 * max_x_shift + 1, 1)).T

    R_map = np.empty(x_shift_arr.shape)

    for (i, j), _ in np.ndenumerate(R_map):
        x_shift = x_shift_arr[i, j]
        y_shift = y_shift_arr[i, j]

        # determine the take positions within the bin for next timestep
        i_tl = max(0, -1 * y_shift)
        j_tl = max(0, -1 * x_shift)
        i_th = M - y_shift if y_shift > 0 else M
        j_th = N - x_shift if x_shift > 0 else N

        # determine the put positions for the shifted next timestep
        i_pl = max(0, y_shift)
        j_pl = max(0, x_shift)
        i_ph = y_shift if y_shift < 0 else M
        j_ph = x_shift if x_shift < 0 else N

        shifted_fluc_b = np.zeros((M, N))

        shifted_fluc_b[i_pl:i_ph, j_pl:j_ph] = fluc_b[i_tl:i_th, j_tl:j_th]
        R_map[i, j] = 1 / (M*N) * np.sum(fluc_a * shifted_fluc_b)

    R_max_i, R_max_j = np.unravel_index(np.argmax(R_map), R_map.shape)
    delta_i = R_max_i - max_y_shift
    delta_j = R_max_j - max_x_shift

    if plot:
        plt.imshow(R_map)
        x_ticks = np.arange(-1*max_x_shift, max_x_shift+1)
        plt.xticks(np.arange(len(x_ticks)), x_ticks)
        y_ticks = np.arange(-1*max_y_shift, max_y_shift+1)
        plt.yticks(np.arange(len(y_ticks)), y_ticks)
        plt.colorbar()
        plt.show()

    return -1 * delta_i, -1 * delta_j


