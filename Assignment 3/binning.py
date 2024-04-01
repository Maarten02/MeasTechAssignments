import matplotlib.pyplot as plt
import numpy as np
from cross_correlate import find_pixel_shift
from tifffile import imread
import warnings


def find_velocity_field(image_a, image_b, binsize, overlap):

    # divide image a and image b into bins
    N_a, M_a = image_a.shape
    N_b, M_b = image_b.shape

    if abs(N_a - N_b) > 1e-6 or abs(M_a - M_b) > 1e-6:
        raise Exception('Images not equal size')

    bin_N = binsize
    bin_M = binsize

    N_spacing = (1 - overlap) * bin_N
    M_spacing = (1 - overlap) * bin_M

    if N_a % N_spacing != 0 or M_a % M_spacing != 0:
        warnings.warn(f'cannot divide ({N_a}, {M_a}) into boxes of ({N_spacing}, {M_spacing})')

    # loop over the bins in a and the bins in b
    x_pos_vel = np.arange(bin_M/2, M_a-bin_M/2, M_spacing)
    y_pos_vel = np.arange(bin_N/2, N_a-bin_N/2, N_spacing)
    # and determine for each bin the pixel shift from a to b

    u_arr = np.empty((len(y_pos_vel), len(x_pos_vel)))
    v_arr = np.empty((len(y_pos_vel), len(x_pos_vel)))

    for i, y in enumerate(y_pos_vel):
        for j, x in enumerate(x_pos_vel):

            i_tl = int(y - bin_M/2)
            i_th = int(y + bin_M/2)
            j_tl = int(x - bin_N/2)
            j_th = int(x + bin_N/2)

            bin_a = image_a[i_tl:i_th, j_tl:j_th]
            bin_b = image_b[i_tl:i_th, j_tl:j_th]
            u, v = find_pixel_shift(bin_a, bin_b)

            u_arr[i, j] = u
            v_arr[i, j] = v

    return u_arr, v_arr, x_pos_vel, y_pos_vel

if __name__ == "__main__":
    im_a = imread('data/test_a.tif')
    im_b = imread('data/test_b.tif')

    im_a = np.dot(im_a[..., :3], [0.299, 0.587, 0.114])
    im_b = np.dot(im_b[..., :3], [0.299, 0.587, 0.114])

    im_a = np.flipud(im_a)
    im_b = np.flipud(im_b)

    u_arr, v_arr, x_pos_vel, y_pos_vel = find_velocity_field(im_a, im_b)
    X, Y = np.meshgrid(x_pos_vel, y_pos_vel)
    plt.quiver(x_pos_vel, y_pos_vel, u_arr, v_arr)

    plt.show()
