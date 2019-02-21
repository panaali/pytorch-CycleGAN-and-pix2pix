import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
from scipy.misc import bytescale
import numpy as np

class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index % len(self.AB_paths)]
        AB = Image.open(AB_path)
        if self.input_nc != 1:  # convert to RGB if the image is not in grayscale format
            AB = AB.convert('RGB')
        else:
            AB = Image.fromarray(bytescale(np.array(AB))) # convert any (incl. 16-bit) image to 8-bit image

        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        transform_params = get_params(self.opt, A.size)
        if self.opt.augment_dataset:
            transform_params['flip'] = False

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        # augmented the dataset
        if self.opt.augment_dataset:
            rotation_mode = int(index / len(self.AB_paths)) % 4
            A.rotate(90 * rotation_mode)
            B.rotate(90 * rotation_mode)
            if index > len(self) / 2:
                A = A.transpose(Image.FLIP_TOP_BOTTOM)
                B = B.transpose(Image.FLIP_TOP_BOTTOM)

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        if self.opt.augment_dataset:
            return len(self.AB_paths) * 8 # if we augment the dataset with all rotation and flips we will get 7 new images.
        return len(self.AB_paths)
