import numpy as np
import matplotlib.pyplot as plt
from binning import find_velocity_field


class Plotting:

    def __init__(self, im_a, im_b, bin_n, ov):

        self.im_a = im_a
        self.im_b = im_b
        self.bin_n = bin_n
        self.ov = ov

        im_a = np.flipud(im_a)
        im_b = np.flipud(im_b)

        self.u_arr, self.v_arr, self.x_pos_vel, self.y_pos_vel = find_velocity_field(im_a, im_b, self.bin_n, self.ov)
        self.X, self.Y = np.meshgrid(self.x_pos_vel, self.y_pos_vel)

        self.fig, (self.vel_ax, self.vor_ax) = plt.subplots(1, 2)
        self.fig.set_size_inches(10, 5)
        self.fig.suptitle(f'binsize = {self.bin_n} pix, overlap = {self.ov * 100}%')

    def plot_velocity(self):

        self.vel_ax.quiver(self.X, self.Y, self.u_arr, self.v_arr)
        self.vel_ax.set_title(f'Velocity')
        self.vel_ax.set_aspect('equal')

    def plot_vorticity(self):

        omega_z_arr = np.empty((self.u_arr.shape[0] - 2, self.u_arr.shape[1] - 2))

        for (i, j), _ in np.ndenumerate(omega_z_arr):
            # start one point away from the boundary
            uv_i = i + 1
            uv_j = j + 1

            dvdx = (self.v_arr[uv_i, uv_j + 1] - self.v_arr[uv_i, uv_j - 1]) / 32
            dudy = (self.u_arr[uv_i + 1, uv_j] - self.u_arr[uv_i - 1, uv_j]) / 32

            omega_z_arr[i, j] = dvdx - dudy

        im = self.vor_ax.imshow(omega_z_arr, origin='lower')

        # x_ticks = self.x_pos_vel[1:-1]
        # y_ticks = self.y_pos_vel[1:-1]
        #
        # # Set x and y ticks on the axis
        # self.vor_ax.set_xticks(np.arange(len(x_ticks)))
        # self.vor_ax.set_xticklabels(x_ticks)
        # self.vor_ax.set_yticks(np.arange(len(y_ticks)))
        # self.vor_ax.set_yticklabels(y_ticks)

        self.fig.colorbar(im, ax=self.vor_ax)
        self.vor_ax.set_title('vorticity in the z direction')
        self.vor_ax.tick_params(axis='x', rotation=45)

    def save(self, tit=''):
        self.plot_velocity()
        self.plot_vorticity()
        self.fig.tight_layout()
        self.fig.savefig(f'figures/{tit}_vorvel_{self.bin_n}_{int(self.ov * 100)}.pdf')