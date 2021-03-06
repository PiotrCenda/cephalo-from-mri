import numpy as np
import scipy.ndimage as nd
from scipy import optimize
from skimage.segmentation import flood
from skimage.morphology import remove_small_objects, remove_small_holes, closing, disk
from skimage.measure import label, regionprops
from skimage.feature import canny
from data_manipulation import timer_block, save_tif


def rotation_matrix_x(theta):
    """
    returns rotation matrix for axis,
    theta should be in radians
    """
    rot_mat = np.array([[1, 0, 0, 0],
                        [0, np.cos(theta), np.sin(theta), 0],
                        [0, -np.sin(theta), np.cos(theta), 0],
                        [0, 0, 0, 1]])
    return rot_mat


def rotation_matrix_y(theta):
    """
    returns rotation matrix for axis,
    theta should be in radians
    """
    rot_mat = np.array([[np.cos(theta), 0, -np.sin(theta), 0],
                        [0, 1, 0, 0],
                        [np.sin(theta), 0, np.cos(theta), 0],
                        [0, 0, 0, 1]])
    return rot_mat


def rotation_matrix_z(theta):
    """
    returns rotation matrix for axis
    theta should be in radians
    """
    rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                        [np.sin(theta), np.cos(theta), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    return rot_mat


def translate_matrix(x, y, z, sx, sy, sz):
    """
    returns rotation matrix for axis
    theta should be in radians
    """
    trans_mat = np.array([[sx, 0, 0, x],
                          [0, sy, 0, y],
                          [0, 0, sz, z],
                          [0, 0, 0, 1]])

    return trans_mat


def center_matrix(transform, shape):
    x_mid = int((shape[1] - 1) / 2)
    y_mid = int((shape[0] - 1) / 2)
    z_mid = int((shape[2] - 1) / 2)

    a = np.array([[1, 0, 0, x_mid],
                  [0, 1, 0, y_mid],
                  [0, 0, 1, z_mid],
                  [0, 0, 0, 1]]).reshape(4, 4)
    return a @ transform @ np.linalg.pinv(a)


def axises_rotations_matrix(theta1, theta2, theta3):
    return rotation_matrix_x(theta1) @ rotation_matrix_y(theta2) @ rotation_matrix_z(theta3)


def rigid_transform(img, args):
    alpha, beta, gamma, x, y, z, sx, sy, sz = args[0], args[1], args[2], args[3], args[4], args[5], \
                                              args[6], args[8], args[7]

    # coordinates for 3d image
    grid_x, grid_y, grid_z = np.meshgrid(np.arange(img.shape[1]),
                                         np.arange(img.shape[0]),
                                         np.arange(img.shape[2]))
    new_grid_x = np.ndarray.flatten(grid_x)
    new_grid_y = np.ndarray.flatten(grid_y)
    new_grid_z = np.ndarray.flatten(grid_z)
    m_x = np.array([new_grid_x,
                    new_grid_y,
                    new_grid_z, np.ones(new_grid_x.shape)])

    # rotate matrix
    transform_rotation_matrix = axises_rotations_matrix(alpha, beta, gamma) @ translate_matrix(x, y, z, sx, sy, sz)
    centered_transform_rotation_matrix = center_matrix(transform_rotation_matrix, img.shape)

    # calculate new coordinates
    m_x_transformed = np.linalg.inv(centered_transform_rotation_matrix) @ m_x

    trans_grid_x = m_x_transformed[1].reshape(grid_x.shape)
    trans_grid_y = m_x_transformed[0].reshape(grid_y.shape)
    trans_grid_z = m_x_transformed[2].reshape(grid_z.shape)
    trans_grids = np.array([trans_grid_x, trans_grid_y, trans_grid_z])
    grid = trans_grids
    transformed_image = nd.map_coordinates(img, grid, order=0)
    return transformed_image


def model_to_register_fitting(image, flood_thresh=0.05):
    median = np.array([nd.median_filter(img, footprint=disk(2)) for img in image]).astype(np.float64)
    model = np.array([flood(img, (0, 0), tolerance=flood_thresh) for img in median])
    closed = np.array([closing(img, disk(5)) for img in model])
    remove_noise = np.array([remove_small_holes(img, area_threshold=1500) for img in closed])
    remove_noise2 = np.array([remove_small_objects(img, min_size=1500) for img in remove_noise])
    return np.array([canny(img, sigma=2) for img in remove_noise2]).astype(np.bool_)


def ssd(a, b):
    err = np.logical_and(a[5:-5, 5:-5, 5:-5], b[5:-5, 5:-5, 5:-5]).astype(np.int64)
    cost = -np.sqrt(np.sum([img.ravel() for img in err]))
    print(f"Cost function: {-cost}")
    return cost


def register_image(image_model, image_to_change):
    start_params = np.array([2.01109052e-04, 1.57808256e-06, 3.65095064e-05, 3.50697591e-04, 2.56535195e-04,
                             -2.36831914e-04, 9.40511337e-01, 9.38207923e-01, 1.00130253e+00])
    # [2.77076163e-03, -7.52885706e-03, 7.03755373e-04, 3.79097329e-01, -2.86304089e-03, 6.44776348e-01,
    #  9.39824479e-01, 9.40039058e-01, 1.00105912e+00]

    def cost_function(params):
        image_changed = rigid_transform(image_to_change, params)
        print(f"Checking parameters: {params}")
        return ssd(image_changed, image_model)

    best_parameters = optimize.minimize(fun=cost_function, x0=start_params)

    return best_parameters.x


def auto_t1_t2_fitting(img):
    print(f"\nAuto fitting started, making models for optimalization...")

    t1_m = model_to_register_fitting(img.t1, flood_thresh=0.05)
    t2_m = model_to_register_fitting(img.t2, flood_thresh=0.03)
    save_tif(t1_m, img_name="t1_fit_before", folder="auto_fitting")
    save_tif(t2_m, img_name="t2_fit_before", folder="auto_fitting")

    with timer_block('t2 to t1 fitting'):
        params = register_image(image_model=t1_m, image_to_change=t2_m)

    print(f"\nFinal rigid parameters: {params}")

    img.t2_rigid_transform(parameters=params)

    t2_m1 = model_to_register_fitting(img.t2, flood_thresh=0.03)
    save_tif(t2_m1, img_name="t2_fit_after", folder="auto_fitting")
