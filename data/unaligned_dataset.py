import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
from scipy.misc import bytescale
import numpy as np
import random
import torch

class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        params = None
        if self.opt.augment_dataset:
            params = {'flip': False}
        self.transform_A = get_transform(self.opt, grayscale=(self.input_nc == 1), params=params)
        self.transform_B = get_transform(self.opt, grayscale=(self.output_nc == 1), params=params)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path)
        B_img = Image.open(B_path)


        if self.input_nc != 1:  # convert to RGB if the image is not in grayscale format
            A_img = A_img.convert('RGB')
        else:
            A_img = Image.fromarray(bytescale(np.array(A_img))) # convert any (incl. 16-bit) image to 8-bit image

        if self.output_nc != 1:
            B_img = B_img.convert('RGB')
        else:
            B_array = np.array(B_img)
            # B_array = np.array(B_img.convert('LA'))[:,:,0]
            #convert image to 0-1
            # b255 = np.ones(B_array.shape) * 255
            # B_array = np.where(B_array < 250, 0, b255)
            # convert edges to 0
            # B_array_original = np.array(B_img)
            # for i in range(1, len(B_array) - 1):
            #     for j in range(1, len(B_array[0]) - 1):
            #         if B_array_original[i][j] != 0:
            #             if B_array_original[i-1][j] != 0 and B_array_original[i-1][j] != B_array_original[i][j]:
            #                 B_array[i][j] = 0
            #             elif B_array_original[i+1][j] != 0 and B_array_original[i+1][j] != B_array_original[i][j]:
            #                 B_array[i][j] = 0
            #             elif B_array_original[i][j-1] != 0 and B_array_original[i][j-1] != B_array_original[i][j]:
            #                 B_array[i][j] = 0
            #             elif B_array_original[i][j+1] != 0 and B_array_original[i][j+1] != B_array_original[i][j]:
            #                 B_array[i][j] = 0
            # B_array = np.where(B_array != 0, 1, 0)    # convert gray scale to 0-1 scale
            B_array = bytescale(B_array)
            B_img = Image.fromarray(B_array) # convert any (incl. 16-bit) image to 8-bit image

        # augmentd the dataset
        if self.opt.augment_dataset:
            A_rotation_mode = int(index / self.A_size) % 4
            B_rotation_mode = int(index / self.B_size) % 4
            # if index > len(self) / 2:
            #     A_img = (get_augment_transform(rotatation_degree=90 * A_rotation_mode, flip=True))(A_img)
            #     B_img = (get_augment_transform(rotatation_degree=90 * B_rotation_mode, flip=True))(B_img)
            # else:
            #     A_img = (get_augment_transform(rotatation_degree=90 * A_rotation_mode))(A_img)
            #     B_img = (get_augment_transform(rotatation_degree=90 * B_rotation_mode))(B_img)
            A_img.rotate(90 * A_rotation_mode)
            B_img.rotate(90 * B_rotation_mode)
            if index > len(self) / 2:
                A_img = A_img.transpose(Image.FLIP_TOP_BOTTOM)
                B_img = B_img.transpose(Image.FLIP_TOP_BOTTOM)


        # apply image transformation
        A = self.transform_A(A_img)
        # assuming the median of the edges of the image is the background color
        # backgroundValue = torch.median(torch.cat((A[0][0], A[0][len(A[0]) - 1], A[0][:, 0], A[0][:, len(A[0][:, 0]) - 1]), 0))
        # A = A.add(backgroundValue.neg()).div(1 + backgroundValue.abs())
        B = self.transform_B(B_img)


        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if self.opt.augment_dataset:
            return max(self.A_size, self.B_size) * 8 # if we augment the dataset with all rotation and flips we will get 7 new images.
        return max(self.A_size, self.B_size)
