import argparse
import glob
import os
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from pathlib import Path


def image_filtering(img_array):
    """Put each object in the image into seperate image, returns list of images """
    objects_array = []
    object_ids = set(np.unique(img_array))
    object_ids -= {0}
    for object_id in object_ids:
        obj_array = np.where(img_array == object_id, img_array, 0)
        objects_array.append(obj_array)
    return objects_array

def generate_edge_file(filepath, folder_path_new):
    img_array = np.array(Image.open(filepath))
    cells_array = image_filtering(img_array)
    dtype= 'uint8'
    final_array = np.zeros(cells_array[0].shape, dtype=dtype)
    for cell_img in cells_array:
        dilated = ndimage.binary_dilation(cell_img).astype(dtype)
        eroded = ndimage.binary_erosion(cell_img).astype(dtype)
        edge = dilated - eroded
        final_array += edge

    final_array2 = np.where(final_array == 0, 0, 255).astype(dtype)
    img_edges = Image.fromarray(final_array2)
    file_name = Path(filepath).name
    img_edges.save(folder_path_new + file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment segmented images.')
    parser.add_argument('--dir', type=str, default='~/cellchallenge/unpaired/Fluo-C2DL-MSC/trainB/', help='directory of the segmented images')
    args = parser.parse_args()
    folder_path = args.dir

    # Create edges folder
    folder_path_new = folder_path + '/../edges/'
    if not os.path.exists(folder_path_new):
        os.makedirs(folder_path_new)

    for filepath in glob.glob(folder_path + '/*.tif'):
        generate_edge_file(filepath, folder_path_new)