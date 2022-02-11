import os
import numpy as np

from data_loader import read_data_from_folder
from data_manipulation import timer_block
from data_plotting import plot_3d
from interpolation import cephalo, interpolate

# parameters calculated by auto fitting function
params_auto = np.array([2.77076163e-03, -7.52885706e-03, 7.03755373e-04, 3.79097329e-01, -2.86304089e-03,
                        6.44776348e-01, 9.39824479e-01, 9.40039058e-01, 1.00105912e+00])


if __name__ == '__main__':
    img = read_data_from_folder(os.path.abspath('data/head'))
    img.t2_rigid_transform(parameters=params_auto)

    with timer_block("cephalometry reconstruction"):
        bones = img.bones_mask()
        soft = img.soft_tissues()
        soft_interpolated = interpolate(soft)
        bones_interpolated = interpolate(bones)
        cephalo(bones_interpolated, soft_interpolated)
        plot_3d(bones_interpolated)
