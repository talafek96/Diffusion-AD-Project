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
    print(rgb.shape)  # TODO: DELETE

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

    size_y, size_x, num_channels = im_tensor.shape[0], im_tensor.shape[1], im_tensor.shape[2]
    print(f"get_sliced_bands_of_HDR_as_256x256_tensor im_tensor.shape = {im_tensor.shape}")  #TODO: DELETE

    assert num_channels == 3

    center_x, center_y = center
    
    lower_x_slice = int(center_x - 256/2)
    upper_x_slice = int(center_x + 256/2)

    lower_y_slice = int(center_y - 256/2)
    upper_y_slice = int(center_y + 256/2)

    if lower_x_slice < 0 or upper_x_slice > size_x or lower_y_slice < 0 or upper_y_slice > size_y:
        raise OutOfBoundsError(f"You have to select a center point that is at least 256/2 pixels from the bounds of the image\n\
                               given (center_x, center_y) = {(center_x, center_y)}, (size_x, size_y) = {(size_x, size_y)}")
    
    sliced_im_tensor = im_tensor[lower_y_slice: upper_y_slice, lower_x_slice: upper_x_slice, :]

    return sliced_im_tensor


def get_next_center_point_of_spectral_image(spectral_im_tensor: th.Tensor):
    """
    Usage:
    >>>  spectral_im_tensor = read_bands_of_HDR_as_tensor(list_of_bands, spectral_image_path)
    >>>  for current_center in get_next_center_point_of_spectral_image(spectral_im_tensor):
    >>>      logic(current_center)
    >>>      # e.g:
    >>>      heatmap, anomaly_score = find_anomalies(list_of_bands, spectral_image_path, current_center)
    """
    size_y, size_x, num_channels = spectral_im_tensor.shape[0], spectral_im_tensor.shape[1], spectral_im_tensor.shape[2]
    
    assert num_channels == 3

    # ROW -> y
    # COL -> x
    num_segment_rows = int(np.ceil(size_y / 256.0))
    num_segment_columns = int(np.ceil(size_x / 256.0))

    initial_center_y = 128
    initial_center_x = 128
    current_center_y = initial_center_y
    current_center_x = initial_center_x

    for row_index in range(num_segment_rows):
        for column_index in range(num_segment_columns):
            current_center_y = min(initial_center_y + 256 * row_index, size_y - 128)
            current_center_x = min(initial_center_x + 256 * column_index, size_x - 128)
            
            yield (current_center_x, current_center_y)


# # Example usage:
# # Assuming 'segments' is a list of torch.Tensor segments
"""
with shape [3, 256, 256]
>>> size_x = 1024  # Original image width
>>> size_y = 768   # Original image height
>>> stitched_result = stitch_segments(segments, size_x, size_y)
"""
def stitch_segments(segments, size_x, size_y):
    """
    Stitches image segments back into the original tensor.

    Args:
        segments (list of torch.Tensor): List of image segments (shape: [3, 256, 256]).
        size_x (int): Original image width.
        size_y (int): Original image height.

    Returns:
        torch.Tensor: Stitched image tensor (shape: [3, size_x, size_y]).
    """
    # Initialize an empty tensor for the stitched image
    stitched_image = th.zeros(size_x, size_y)

    # Calculate the number of segments in each dimension
    num_segment_columns = int(np.ceil(size_x / 256.0))
    num_segment_rows = int(np.ceil(size_y / 256.0))

    # # Iterate over segments and copy content into the stitched image
    # for i in range(num_segments_x):
    #     for j in range(num_segments_y):
    #         segment = segments[i * num_segments_y + j]
    #         stitched_image[:, i * 256 : (i + 1) * 256, j * 256 : (j + 1) * 256] = segment
    for row_index in range(num_segment_rows):
        for col_index in range(num_segment_columns):
            segment = segments[col_index * num_segment_rows + row_index]

            # If this is the last row
            if row_index == num_segment_rows - 1:
                min_row, max_row = size_y - 256, size_y
            else:
                min_row, max_row = row_index * 256, (row_index+1) * 256

            # If this is the last column
            if col_index == num_segment_columns - 1:
                min_col, max_col = size_x - 256, size_x
            else:
                min_col, max_col = col_index * 256, (col_index+1) * 256
                
            assert max_row - min_row == 256
            assert max_col - min_col == 256
            # "[:" to select all channels
            # from min_row to max_row
            # and from min_col to max_col
            stitched_image[min_col : max_col, min_row : max_row] = segment

    return stitched_image


def main():
    HDR_swir = envi.open(HDR_PATH_SWIR, IMAGE_PATH_SWIR)
    # im_swir_rgb = show_img(HDR_swir)  # differnt band in each chunnel
    im_swir_single_band = show_bands([10, 20, 40], HDR_swir)# same band in each chunnel
    print(get_sliced_bands_of_HDR_as_256x256_tensor([1, 10, 20], IMAGE_PATH_SWIR, (460, 220)))
    plt.imshow(get_sliced_bands_of_HDR_as_256x256_tensor([1, 10, 20], IMAGE_PATH_SWIR, (460, 220)))
    plt.show()


if __name__ == '__main__':
    main()
