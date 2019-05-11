"""
This file will create new augmented images based on the segmented image.
each segmented image should be a .tif file and each object in the image should have unique value in the image (e.g. 2)
The algorithm will rotate each object and place it randomly and makes sure that: 
    - No two object will have overlap.
    - Trying to fit all the objects in the result. In rare circumstances after 10000 trying maybe it skip that object.
        This may happen for the last object. 
    - Some objects may touch each other or the edge of the image but not as much as it happens in the original images.


"""


from PIL import Image
from scipy.misc import bytescale
from random import randrange
import numpy as np
import os
import glob
import pathlib

folder_path = '/Users/aliakbarpanahi/Documents/00-Jobs/GAN/segmentation/DIC/B/train/'
number_of_new_files = 100


def augment_image(filepath, folder_path_new, number_of_new_files):
    def image_filtering(img_array):
        """Put each object in the image into seperate image, returns list of images """
        objects_array = []
        for object_id in np.unique(img_array):
            obj_array = np.where(img_array == object_id, img_array, 0)
            objects_array.append(obj_array)
        return objects_array

    filename, file_extension = pathlib.Path(filepath).stem, pathlib.Path(filepath).suffix
    img = Image.open(filepath)
    img_array = np.array(img)

    # save original image
    img_scaled = Image.fromarray(bytescale(img_array))
    img_scaled.save(folder_path_new + filename + '_0'+ file_extension)


    objects_array = image_filtering(img_array)

    # Generate 100 new image:
    for i in range(number_of_new_files):
        background_array = objects_array[0].copy()
        for j in range(1, len(objects_array)):
            object_array = objects_array[j]
            object_image = Image.fromarray(object_array)
            object_image_scaled = Image.fromarray(bytescale(object_array))
            object_bounding_box = object_image_scaled.getbbox()
            object_image_croped = object_image.crop(object_bounding_box)
            newbackground = np.full(background_array.shape, len(objects_array))
            while_counter = 0
            while max(np.unique(background_array + newbackground)) != max(np.unique(object_image_croped)):
                while_counter += 1
                if while_counter == 10000:
                    newbackground = np.zeros(background_array.shape)
                    break
                angle = randrange(0, 360)
                object_image_croped_rotated = object_image_croped.rotate(angle, expand=True)

                upper_left_x = randrange(0, object_array.shape[0] - int(object_image_croped.width))  #  # to prevent pasting on the edge
                upper_left_y = randrange(0, object_array.shape[1] - int(object_image_croped.height))  #

                newbackground = Image.fromarray(
                    np.zeros((objects_array[0].shape[0], objects_array[0].shape[1]), dtype='uint8'))
                newbackground.paste(object_image_croped_rotated, (upper_left_x, upper_left_y))

            background_array = background_array + newbackground

        background = Image.fromarray(bytescale(background_array))
        background.save(folder_path_new + filename  + '_' + str(i + 1) + file_extension)


if __name__ == '__main__':
    # Create augmented folder
    folder_path_new = folder_path + 'augmented/'
    if not os.path.exists(folder_path_new):
        os.makedirs(folder_path_new)

    # Read tif files in the folder_path and for each generate number_of_new_files(e.g. 100) new image in the folder_path_new
    for filepath in glob.glob(folder_path + '*.tif'):
        augment_image(filepath, folder_path_new, number_of_new_files)


