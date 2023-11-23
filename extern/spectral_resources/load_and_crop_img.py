import spectral.io.envi as envi
import spectral
# from spectral import 
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import os
from typing import List, Tuple


# # SWIR 50m
# HDR_PATH_SWIR = r"extern/spectral_resources/raw_15451_rdk_rd_rf.hdr"
# IMAGE_PATH_SWIR = r"extern/spectral_resources/raw_15451_rdk_rd_rf"

# SWIR 200m
HDR_PATH_SWIR = os.path.abspath("extern/spectral_resources/raw_7687_rdk_rd_rf.hdr")
IMAGE_PATH_SWIR = os.path.abspath("extern/spectral_resources/raw_7687_rdk_rd_rf")


class OutOfBoundsError(ValueError):
    pass


def show_bands(list_of_bands, HDR):
    if len(list_of_bands) == 1:
        rgb = np.array(HDR.read_band(list_of_bands[0]))
    else:
        rgb = np.stack([HDR.read_band(list_of_bands[0]), HDR.read_band(list_of_bands[1]),
                        HDR.read_band(list_of_bands[2])], axis=-1)
        debug = th.from_numpy(rgb)
    # rgb = rgb / rgb.max() * 1.5

    figure = plt.figure()
    figure.canvas.manager.set_window_title("RGB Image")
    plt.imshow(rgb)
    plt.show()
    return rgb


def read_bands_of_HDR_as_tensor(list_of_bands: List, image_path: str):
    #HDR_swir = envi.open(HDR_PATH_SWIR, IMAGE_PATH_SWIR)
    header_path = "%s.hdr" % image_path
    HDR_swir = envi.open(header_path, image_path)
    
    if len(list_of_bands) == 1:
        rgb = np.array(HDR_swir.read_band(list_of_bands[0]))
    else:
        rgb = np.stack([HDR_swir.read_band(list_of_bands[0]), HDR_swir.read_band(list_of_bands[1]),
                        HDR_swir.read_band(list_of_bands[2])], axis=-1)
    
    return th.from_numpy(rgb)


def get_sliced_bands_of_HDR_as_256x256_tensor(list_of_bands: List, image_path: str, center: Tuple[int, int]):
    im_tensor = read_bands_of_HDR_as_tensor(list_of_bands, image_path)

    size_x, size_y, num_channels = im_tensor.shape[0], im_tensor.shape[1], im_tensor.shape[2]

    assert num_channels == 3

    center_x, center_y = center
    
    lower_x_slice = int(center_x - 256/2)
    upper_x_slice = int(center_x + 256/2)

    lower_y_slice = int(center_y - 256/2)
    upper_y_slice = int(center_y + 256/2)

    if lower_x_slice < 0 or upper_x_slice > size_x or lower_y_slice < 0 or upper_y_slice > size_y:
        raise OutOfBoundsError("You have to select a center point that is at least 256/2 pixels from the bounds of the image")
    
    sliced_im_tensor = im_tensor[lower_x_slice: upper_x_slice, lower_y_slice: upper_y_slice, :]

    return sliced_im_tensor


def main():
    HDR_swir = envi.open(HDR_PATH_SWIR, IMAGE_PATH_SWIR)
    # im_swir_rgb = show_img(HDR_swir)  # differnt band in each chunnel
    im_swir_single_band = show_bands([10, 20, 40], HDR_swir)# same band in each chunnel
    print(get_sliced_bands_of_HDR_as_256x256_tensor([1, 10, 20], IMAGE_PATH_SWIR, (510, 130)))


if __name__ == '__main__':
    main()
