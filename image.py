import numpy as np
from skimage.segmentation import flood
from skimage.morphology import remove_small_holes, remove_small_objects, disk, diamond, ball, closing, dilation, erosion
from skimage.measure import label, regionprops
from skimage.filters.rank import mean_bilateral
from skimage.util import img_as_ubyte
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from itertools import repeat

from data_rigid_transform import rigid_transform
from data_manipulation import func_timer, save_tif


class ImageSequences:
    """
    Class where T1 and T2 sequences are stored. Additionally there are functions which thresholds, masks and segments
    objects on images.
    """

    def __init__(self, img_dict):
        self.__all = img_dict
        self.__t1 = img_dict['T1']
        self.__t2 = img_dict['T2']
        self.__shape = img_dict['T1'][0].shape

    def __copy__(self, data_dict=None):
        copy = ImageSequences(self.__all)
        return copy

    @property
    def t1(self):
        return self.__t1

    @property
    def t2(self):
        return self.__t2

    @property
    def shape(self):
        return self.__shape

    def t2_rigid_transform(self, parameters):
        self.__t2 = rigid_transform(self.__t2, parameters)

    @func_timer
    def background_mask(self):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            p1 = pool.map_async(mean_bilateral_wrap, [image for image in self.__t1])
            p2 = pool.map_async(mean_bilateral_wrap, [image for image in self.__t2])

            t1 = p1.get()
            t2 = p2.get()

            t1 = np.array(((t1 - np.min(t1)) / np.ptp(t1))).astype(np.float64)
            t2 = np.array(((t2 - np.min(t2)) / np.ptp(t2))).astype(np.float64)

            p1_2 = pool.map_async(flood_wrap, [image for image in t1])
            p2_2 = pool.map_async(flood_wrap, [image for image in t2])

            and_img = np.logical_and(p1_2.get(), p2_2.get())

            remove = pool.map(remove_wrap, [image for image in and_img])
            eroded = pool.starmap(erosion, zip(remove, repeat(diamond(5))))

            size = max([region.area for region in regionprops(label(np.array(eroded), connectivity=3))] + [1])
            remove_2 = remove_small_objects(np.array(eroded), min_size=(size - 1), connectivity=3)

            closed = pool.starmap(closing, zip(remove_2, repeat(disk(11))))
            dilated = pool.starmap(dilation, zip(closed, repeat(disk(11))))

            and_img_2 = np.logical_and(remove, dilated)

            remove_3 = pool.map(remove_biggest_hl, [image for image in and_img_2])
            closed_2 = pool.starmap(closing, zip(remove_3, repeat(disk(15))))
            dilated_2 = pool.starmap(dilation, zip(closed_2, repeat(disk(9))))
            result = pool.starmap(closing, zip(dilated_2, repeat(disk(15))))

        return result

    @func_timer
    def soft_tissues(self):
        with mp.Pool(processes=mp.cpu_count()) as pool:
            p1 = pool.map_async(mean_bilateral_wrap2, [image for image in self.__t1])
            p2 = pool.map_async(mean_bilateral_wrap2, [image for image in self.__t2])
            t1 = p1.get()
            t2 = p2.get()

            t1 = np.array(((t1 - np.min(t1)) / np.ptp(t1))).astype(np.float64)
            t2 = np.array(((t2 - np.min(t2)) / np.ptp(t2))).astype(np.float64)

            thresh_t1 = t1 >= 0.1
            thresh_t2 = t2 >= 0.2

            p1_2 = pool.map_async(remove_wrap, [image for image in thresh_t1])
            p2_2 = pool.map_async(remove_wrap, [image for image in thresh_t2])
            t1 = np.array(p1_2.get()).astype(np.float64)
            t2 = np.array(p2_2.get()).astype(np.float64)

            or_soft = np.logical_or(t1, t2)

            remove = np.array([remove_wrap(img) for img in or_soft])

            dilated = pool.starmap(dilation, zip(remove, repeat(disk(3))))

        size = max([region.area for region in regionprops(label(np.array(dilated), connectivity=3))] + [1])
        remove_2 = remove_small_objects(np.array(dilated), min_size=(size - 1), connectivity=3)

        result = np.logical_and(remove, remove_2)

        return result

    @func_timer
    def bones_mask(self):
        """
        remove sinuses and air --> erozja, stworzenie obiektów i sprawdzanie maksymalnej średnicy np.
        sprawdzić!!
        """
        with ThreadPool(processes=mp.cpu_count()) as pool:
            background = pool.apply_async(self.background_mask)
            soft = pool.apply_async(self.soft_tissues)

            p1 = pool.map_async(mean_bilateral_wrap3, [image for image in self.__t1])
            p2 = pool.map_async(mean_bilateral_wrap3, [image for image in self.__t2])
            t1_3 = p1.get()
            t2_3 = p2.get()

            t1_3 = np.array(((t1_3 - np.min(t1_3)) / np.ptp(t1_3))).astype(np.float64)
            t2_3 = np.array(((t2_3 - np.min(t2_3)) / np.ptp(t2_3))).astype(np.float64)

            thresh_t1 = t1_3 <= 0.1
            thresh_t2 = t2_3 <= 0.2

            no_soft_tissues = np.logical_not(np.logical_or(background.get(), soft.get()))

            bones = np.logical_and(thresh_t1, thresh_t2)

            p3 = pool.map_async(remove_wrap_bone, [image for image in bones])
            bones_remove = p3.get()

            bones_2 = np.logical_and(bones_remove, no_soft_tissues)

            remove = np.array([remove_small_objects(img, min_size=30) for img in bones_2])

        dilated = dilation(remove, ball(2))

        size = max([region.area for region in regionprops(label(np.array(dilated), connectivity=3))] + [1])
        remove_2 = remove_small_objects(np.array(dilated), min_size=(size - 1), connectivity=3)

        result = np.logical_and(remove, remove_2)

        return result


def remove_wrap(img):
    img = remove_small_holes(img, area_threshold=60)
    return remove_small_objects(img, min_size=10)


def remove_wrap_bone(img):
    img = remove_small_holes(img, area_threshold=20)
    return remove_small_objects(img, min_size=20)


def remove_biggest_obj(img):
    img = np.array(img)
    size = max([region.area for region in regionprops(label(img))] + [1])
    remove = remove_small_objects(img, min_size=(size - 1))
    return remove


def remove_biggest_hl(img):
    img = np.array(img)
    size = max([region.area for region in regionprops(label(np.logical_not(img)))] + [1])
    remove = remove_small_holes(img, area_threshold=(size - 1))
    return remove


def mean_bilateral_wrap(img):
    return mean_bilateral(img_as_ubyte(img), disk(7))


def mean_bilateral_wrap2(img):
    return mean_bilateral(img_as_ubyte(img), disk(3))


def mean_bilateral_wrap3(img):
    return mean_bilateral(img_as_ubyte(img), disk(3))


def flood_wrap(img):
    return flood(img, seed_point=(0, 0), tolerance=0.07)
