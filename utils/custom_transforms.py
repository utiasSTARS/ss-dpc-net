from __future__ import division
import torch
import numpy as np
import random
import torch.utils.data
from scipy.misc import imresize
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
#    def __init__(self, img_mode):
#        self.img_mode = img_mode
    def __call__(self, images, intrinsics, targets):
        tensors = []
        for im in images:
            # put it from HWC to CHW format
#            if self.img_mode == 'L':
#                im = im.reshape((im.shape[0],im.shape[1],1))
            im = np.transpose(im, (2, 0, 1))
            tensors.append(torch.from_numpy(im).float()/255)
        return tensors, intrinsics, targets


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""
    def __init__(self, p=None):
        self.prob = p
    def __call__(self, images, intrinsics, targets):
        assert intrinsics is not None
        if self.prob is not None:
            prob = np.random.uniform(low=0, high=1)
            if prob <= self.prob:
                output_targets = np.copy(targets)
                output_intrinsics = np.copy(intrinsics)
                output_images = [np.copy(np.fliplr(im)) for im in images]
                w = output_images[0].shape[1]
                output_intrinsics[0,2] = w - output_intrinsics[0,2]
                output_targets[5] = -targets[5] #works only for yaw atm
            else:
                output_images = images
                output_intrinsics = intrinsics
                output_targets = targets
          
            return output_images, output_intrinsics, output_targets
        else:
            return images, intrinsics, targets


class RandomScaleCrop(object):
    """Randomly zooms images up to 15% and crop them to keep same size as before.
        TODO: check that target doesn't actually need to change based on the scaling/cropping"""

    def __call__(self, images, intrinsics, targets):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)

        in_h, in_w, _ = images[0].shape
        x_scaling, y_scaling = np.random.uniform(1,1.15,2)
        scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

        output_intrinsics[0] *= x_scaling
        output_intrinsics[1] *= y_scaling
        scaled_images = [imresize(im, (scaled_h, scaled_w)) for im in images]

        offset_y = np.random.randint(scaled_h - in_h + 1)
        offset_x = np.random.randint(scaled_w - in_w + 1)
        cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]

        output_intrinsics[0,2] -= offset_x
        output_intrinsics[1,2] -= offset_y

        return cropped_images, output_intrinsics, targets

class Resize(object):
    def __init__(self, new_dim=(120,400)):
        self.new_dim=new_dim
    def __call__(self, images, intrinsics, targets):
        assert intrinsics is not None
        output_intrinsics = np.copy(intrinsics)
        downscale_y = float(images[0].shape[0])/self.new_dim[0]
        downscale_x = float(images[0].shape[1])/self.new_dim[1]

#        downscale = 370./self.new_dim[0]
        output_intrinsics[:,0] = intrinsics[:,0]/downscale_x
        output_intrinsics[:,1] = intrinsics[:,1]/downscale_y
        resized_imgs = [imresize(im, self.new_dim) for im in images]
#        resized_imgs = images
        return resized_imgs, output_intrinsics, targets

class PILtoNumpy(object):
    def __call__(self, images, intrinsics, targets):
        assert intrinsics is not None
        resized_imgs = []
        for im in images:
            new_arr = np.array(im)
#            new_arr = np.expand_dims(arr, 2)
            resized_imgs.append(new_arr)
        return resized_imgs, intrinsics, targets

class RandomJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    def __call__(self, images, intrinsics, targets):
        brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)

        output_imgs = [F.adjust_hue(F.adjust_saturation(F.adjust_contrast(F.adjust_brightness(im, \
            brightness_factor), contrast_factor), saturation_factor), hue_factor) for im in images]
        return output_imgs, intrinsics, targets
