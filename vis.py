import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision

def canvas_to_array(fig):    
    #Convert matplotlib figure to a C X H X W numpy array for TensorboardX
    canvas = fig.canvas
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image_np = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height.astype(np.uint32), width.astype(np.uint32), 3)
    #PIL expects CXHXW
    return np.rollaxis(image_np, 2)

def plot_disp(disp, save_file = None):
    fig = plt.figure(1)
    ax = plt.gca()
    ax.imshow(np.log((disp/0.2 )+ 1) , cmap='plasma')
#    ax.imshow(np.log(disp/disp.max()+1), cmap='plasma')
#    depth = 1.0/disp
#    ax.imshow(depth, cmap='plasma', vmin=0, vmax=85)
    
    image_array = canvas_to_array(fig)
    if save_file!=None:
        plt.savefig(save_file, dpi=400, bbox_inches='tight')
    plt.close(fig)
    return image_array

def plot_img_array(img_array, nrow=3, save_file = None):
    img_grid = torchvision.utils.make_grid(img_array, nrow=nrow)
    fig = plt.figure(1)
    ax = plt.gca()
    ax.imshow(img_grid.permute(1,2,0))
    image_array = canvas_to_array(fig)
    
    if save_file!=None:
        plt.savefig(save_file, dpi=400, bbox_inches='tight')
        
    plt.close(fig)
    return image_array

def plot_multi_traj(traj1, label_str1, traj2, label_str2, title_str):
#    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
#    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig = plt.figure(1)
    ax = plt.gca()
    
    ax.plot(traj1[:,0,3], traj1[:,1,3], label=label_str1)
    ax.plot(traj2[:,0,3], traj2[:,1,3], '--', label=label_str2)
    ax.set_title(title_str)
    ax.legend()
    
#    fig = plt.figure(ax)
    image_array = canvas_to_array(fig)
    plt.close(fig)
    return image_array

def plot_traj(pred, label_str, title_str):
#    fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
#    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    fig = plt.figure(1)
    ax = plt.gca()
    
    ax.plot(pred[:,0,3], pred[:,1,3], '--', label=label_str)
    ax.set_title(title_str)
    ax.legend()
    
#    fig = plt.figure(ax)
    image_array = canvas_to_array(fig)
    plt.close(fig)
    return image_array

def plot_6_by_1(tau, title=''):
    fig, ax = plt.subplots(6, 1, sharex='col', sharey='row')

    r = tau[:, :3] 
    phi = tau[:, 3:]

    ax[0].plot(r[:, 0], label='x')
    ax[1].plot(r[:, 1], label='y')
    ax[2].plot(r[:, 2], label='z')

    ax[3].plot(phi[:, 0],label='$\Theta_1$')
    ax[4].plot(phi[:, 1], label='$\Theta_2$')
    ax[5].plot(phi[:, 2], label='$\Theta_3$')
    plt.title(title)
    image_array = canvas_to_array(fig)
    plt.close(fig)
    return image_array

def UnNormalize_img_array(tensor):
    B,C,H,W = tensor.size()
    tensor = tensor.cpu()
    transform = transforms.Compose([
        UnNormalize(mean = [0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225]),
        Clamp(0, 1),
#        transforms.ToPILImage()  # multiplication by 255 happens here
    ])
    img_array = transform(tensor)
#    img_array = torch.cat(transform(tensor),0)
#    img_array = img_array.view(B,C,H,W)
    return img_array

class Clamp:
    """Clamp all elements in input into the range [min, max].
    Args:
        min (Number): lower-bound of the range to be clamped to
        min (Number): upper-bound of the range to be clamped to
    """

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): the input Tensor
        Returns:
            Tensor: the result Tensor
        """
        return tensor.clamp(self.min, self.max)
    
class UnNormalize:
    """Scale a normalized tensor image to have mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] * std[channel]) + mean[channel]) ``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (N, C, H, W) to be un-normalized.
        Returns:
            Tensor: Un-normalized Tensor image.
        """
        for i in range(0, tensor.size(0)):
            for t, m, s in zip(tensor[i], self.mean, self.std):
                t.mul_(s).add_(m)

        return tensor









































