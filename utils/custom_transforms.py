from __future__ import division
import torch
import numpy as np
import random
import torch.utils.data
import torchvision.transforms.functional as F

def get_data_transforms(config):
                ## specify transforms for data augmentation.  Only transform the training data and keep test and val data as-is
    if config['normalize_img'] == True:
        mean = [0.485, 0.456, 0.406] #These are ImageNet normalizations
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0,0,0]
        std = [1,1,1]
    data_transforms = {
        'train': Compose([
            ArrayToTensor(),
            Normalize(mean=mean, std=std),
        ]),
        'val': Compose([
            ArrayToTensor(),
            Normalize(mean=mean, std=std),
        ])
    }
    return data_transforms

class tofloatTensor(object): #transform that converts a numpy array to a FloatTensor
    def __init__(self):
        self.x=0
    def __call__(self, input):
        out = torch.FloatTensor(input)#.type(torch.LongTensor)
        return out  

    ###Visual Transforms

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, images, intrinsics, targets):
        for t in self.transforms:
            images, intrinsics, targets = t(images, intrinsics, targets)
        return images, intrinsics, targets


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, images, intrinsics, targets):
        for tensor in images:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
        return images, intrinsics, targets


class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""
    def __call__(self, images, intrinsics, targets):
        tensors = []
        for im in images:
            im = np.transpose(im, (2, 0, 1))
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, intrinsics, targets

class PILtoNumpy(object):
    def __call__(self, images, intrinsics, targets):
        assert intrinsics is not None
        resized_imgs = []
        for im in images:
            new_arr = np.array(im)
            resized_imgs.append(new_arr)
        return resized_imgs, intrinsics, targets

