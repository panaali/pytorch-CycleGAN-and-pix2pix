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
import random
folder_path = '/Users/aliakbarpanahi/Documents/00-Jobs/GAN/segmentation/DIC/B/train/'
number_of_new_files = 100


def generate_image(object_library, number_of_unique_objects, shape):
    final_array = np.zeros(shape, dtype='uint16')
    # same distribution of objects as original dataset
    obj_id = 1
    for j in range(random.choice(number_of_unique_objects)):
        object_array = random.choice(object_library).copy()
        object_array[object_array == 1] = obj_id
        object_image = Image.fromarray(object_array)
        new_background_array = np.full(shape, 100000, dtype='uint16') # ones instead of zeros so make sure we enter the while loop at least once
        while_counter = 0
        def is_overlap():
            m = max(np.unique(final_array + new_background_array))
            res = m != obj_id
            return res
        while is_overlap(): # when have overlap it will have number 2 or more
            while_counter += 1
            if while_counter % 100 == 0:
                print(j, while_counter)
            if while_counter == 1000:
                new_background_array = np.zeros(shape, dtype='uint16') # so we add nothing to the final image
                obj_id -= 1
                break
            angle = randrange(0, 360)
            if angle == 90 or angle == 180 or angle == 270:
                continue
            object_image_croped_rotated = object_image.rotate(angle, expand=True)
            upper_left_x = randrange(0, shape[0] - object_image_croped_rotated.width // 2)# to prevent pasting on the edge less than half
            upper_left_y = randrange(0, shape[1] - object_image_croped_rotated.height // 2)
            new_background_image = Image.fromarray(np.zeros(shape, dtype='uint16'))
            new_background_image.paste(object_image_croped_rotated, (upper_left_x, upper_left_y))
            new_background_array = np.array(new_background_image)

        final_array = final_array + new_background_array
        obj_id += 1

    return Image.fromarray(bytescale(final_array))



def add_to_object_library(filepath, object_library, number_of_unique_objects, folder_path_original_scaled):
    def image_filtering(img_array):
        """Put each object in the image into seperate image, returns list of images """
        objects_array = []
        object_ids = set(np.unique(img_array))
        number_of_unique_objects.append(len(object_ids) - 1)
        # Remove objects on the image edges
            # go through the edge of the image and find all the ids. then remove them from object_ids
        number_of_rows = img_array.shape[0]
        number_of_cols = img_array.shape[1]
        top_edge_ids = set(np.unique(img_array[0, :]))
        bottom_edge_ids = set(np.unique(img_array[number_of_rows - 1, :]))
        left_edge_ids = set(np.unique(img_array[:, 0]))
        right_edge_ids = set(np.unique(img_array[:, number_of_cols-1]))

        object_ids -= {0}
        object_ids -= top_edge_ids
        object_ids -= bottom_edge_ids
        object_ids -= left_edge_ids
        object_ids -= right_edge_ids

        for object_id in object_ids:
            obj_array = np.where(img_array == object_id, img_array, 0)
            objects_array.append(obj_array)
        return objects_array

    filename, file_extension = pathlib.Path(filepath).stem, pathlib.Path(filepath).suffix
    img = Image.open(filepath)
    img_array = np.array(img)

    # save original image
    img_scaled = Image.fromarray(bytescale(img_array))
    img_scaled.save(folder_path_original_scaled + filename + file_extension)

    objects_array = image_filtering(img_array)
    for j in range(0, len(objects_array)):
        object_array = objects_array[j]
        object_image = Image.fromarray(object_array)
        object_image_scaled = Image.fromarray(bytescale(object_array))
        object_bounding_box = object_image_scaled.getbbox()
        object_image_croped = object_image.crop(object_bounding_box)
        object_image_croped_array = np.array(object_image_croped)
        object_image_croped_array[object_image_croped_array != 0] = 1
        object_library.append(object_image_croped_array)

if __name__ == '__main__':
    # Create augmented folder
    folder_path_original_scaled = folder_path + 'scaled/'
    folder_path_new = folder_path + 'augmented/'
    if not os.path.exists(folder_path_original_scaled):
        os.makedirs(folder_path_original_scaled)
    if not os.path.exists(folder_path_new):
        os.makedirs(folder_path_new)
    object_library = []
    number_of_unique_objects = []
    # Read tif files in the folder_path and add objects in the images to the library
    first = True
    for filepath in glob.glob(folder_path + '*.tif'):
        if first:
            img = Image.open(filepath)
            images_shape = (img.height, img.width)
            first = False
        add_to_object_library(filepath, object_library, number_of_unique_objects, folder_path_original_scaled)

    # Generate new images using this library.
    for i in range(number_of_new_files):
        newImage = generate_image(object_library, number_of_unique_objects, images_shape)
        newImage.save(folder_path_new + str(i) + '.tif')

