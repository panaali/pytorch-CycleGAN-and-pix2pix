from PIL import Image
from scipy.misc import bytescale
from random import randrange
import numpy as np

file_path = './dataset/01_t002.tif'
img = Image.open(file_path)
img_array = np.array(img)

def image_filtering(img_array):
    objects_array = []
    for object_id in np.unique(img_array):
        obj_array = np.where(img_array == object_id, img_array, 0)
        objects_array.append(obj_array)
    return objects_array

objects_array = image_filtering(img_array)
skip = True
for object_array in objects_array:
    if skip:
        skip = False
        continue

    object_image_scaled = Image.fromarray(bytescale(object_array))
    object_bounding_box = object_image_scaled.getbbox()
    object_image_croped = object_image_scaled.crop(object_bounding_box)

    angle = randrange(0,360)
    object_image_croped_rotated = object_image_croped.rotate(angle, expand= True)

    background = Image.fromarray(np.zeros((object_array.shape[0], object_array.shape[1]), dtype='uint16'))

    x = randrange(0, object_array.shape[0] - int(object_image_croped.width /2))
    y = randrange(0, object_array.shape[1] - int(object_image_croped.height /2))
    background.paste(object_image_croped_rotated, (x,y))

    background.show()
    object_image_scaled.show()
    # object_image_croped.show()
    # object_image_croped_rotated.show()

    break