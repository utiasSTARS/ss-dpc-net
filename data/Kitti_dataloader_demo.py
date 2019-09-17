from data.kitti_loader import KittiLoaderPytorch
import torch
from utils.learning_helpers import *
from utils.custom_transforms import *
data_dir = '/media/brandon/DATA/KITTI/raw'

config={
    'num_frames': None,
    'skip':1,    ### if not one, we skip every 'skip' samples that are generated ({1,2}, {2,3}, {3,4} becomes {1,2}, {3,4})
    'correction_rate': 1, ### if not one, only perform corrections every 'correction_rate' frames (samples become {1,3},{3,5},{5,7} when 2)
    'img_per_sample': 2,
    'imu_per_sample': (2-1)*10, #skip * (img_per_sample -1)*10
    'minibatch': 32,       ##minibatch size      
    'augment_motion': False, #add more training data where data skips frames to simulate faster motion.
    'normalize-img': False,
    'augment_backwards': False,
    'use_flow': False,
    'load_lidar_depth': False,
    'load_stereo_depth': False,
    'dropout_prob': 0.0,
    }


dsets = KittiLoaderPytorch(data_dir, config, [['00'], ['02']], mode='val', transform_img=get_data_transforms(config)['val'],load_lidar=config['load_lidar_depth'], load_stereo=config['load_stereo_depth'])
dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=config['minibatch'], shuffle=False, num_workers=4)

imgs, imu, labels, delta_t, intrinsics, svo_pose_vec, depth_data = dsets.__getitem__(0)


print(dsets['train'].cam_filenames[0][0], dsets['train'].cam_filenames[0][1])
print(dsets['train'].ts_samples[0], dsets['train'].ts_samples[1])