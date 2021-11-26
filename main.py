import os
import numpy as np

from data_loader import read_data_from_folder
from data_manipulation import save_tif, timer_block
from data_plotting import plot_3d, plot_3d_surface
from interpolation import scale_z_to_y, show_xyz, show_rec_xyz, zy_to_tif

# parameters calculated by auto fitting function
params_auto = np.array([2.01109052e-04, 1.57808256e-06, 3.65095064e-05, 3.50697591e-04, 2.56535195e-04,
                        -2.36831914e-04, 9.40511337e-01, 9.38207923e-01, 1.00130253e+00])

# TODO: update soft tissues and bone masks


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    img.t2_rigid_transform(parameters=params_auto)

    with timer_block("bones mask making with interpolation"):
        bones = img.bones_mask()
        save_tif(bones, img_name="bones_mask", folder="masks")
        scale_z_to_y(bones)
        bones_interpolated = zy_to_tif()
        save_tif(bones_interpolated, img_name='bones_mask_interpolated', folder='masks')
        show_rec_xyz(bones_interpolated)
        plot_3d(bones_interpolated)
