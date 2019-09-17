import torch
import sys
sys.path.insert(0,'..')
from data.kitti_loader import KittiLoaderPytorch
from validate import Validate, test_depth_and_reconstruction, test_trajectory
import models.stn as stn
import models.mono_model_joint as mono_model_joint
from utils.learning_helpers import *
from utils.custom_transforms import *
import paper_plots_and_data.visualizers as visualizers
from pyslam.metrics import TrajectoryMetrics
import numpy as np
import argparse
import os 
import glob
from liegroups import SE3

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device(0)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_dir', type=str, default='/media/brandon/DATA/KITTI-odometry-downsized/')
parser.add_argument('--mode', type=str, default='online')
parser.add_argument('--estimator_type', type=str, default='stereo')
parser.add_argument('--val_seq', nargs='+',type=str, default='19')
parser.add_argument('--exploss', action='store_true', default=True)
config={
    'num_frames': None,
    'skip':1,    ### if not one, we skip every 'skip' samples that are generated ({1,2}, {2,3}, {3,4} becomes {1,2}, {3,4})
    'correction_rate': 1, ### if not one, only perform corrections every 'correction_rate' frames (samples become {1,3},{3,5},{5,7} when 2)
    'img_per_sample': 2,
    'imu_per_sample': (2-1)*10, #skip * (img_per_sample -1)*10
    'minibatch': 1,       ##minibatch size      
    'augment_motion': False, #add more training data where data skips frames to simulate faster motion.
    'normalize_img': True,
    'augment_backwards': False,
    'use_flow': True,
    'dropout_prob': 0.0,
    }

args = parser.parse_args()
for k in args.__dict__:
    config[k] = args.__dict__[k]
    
model_dirs = 'paper_plots_and_data/online-models/'
date = 'best_stereo'
pretrained_path = 'paper_plots_and_data/online-models/best_stereo/2019-6-24-13-4-most_loop_closures-val_seq-00-test_seq-05.pth'

output_dir = '{}{}/'.format(model_dirs,date)
args.data_dir = '{}/{}'.format(args.data_dir,args.mode)
seq = [args.val_seq] #model.replace(output_dir,'').replace('/','').replace
figures_output_dir = '{}figs'.format(output_dir)
os.makedirs(figures_output_dir,exist_ok=True)
os.makedirs(figures_output_dir+'/imgs', exist_ok=True)
os.makedirs(figures_output_dir+'/depth', exist_ok=True)
os.makedirs(figures_output_dir+'/exp_mask', exist_ok=True)

test_dset = KittiLoaderPytorch(args.data_dir, config, [seq, seq, seq], mode='test', transform_img=get_data_transforms(config)['val'])
test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=4)
eval_dsets = {'test': test_dset_loaders}
Reconstructor = stn.Reconstructor().to(device)
model = mono_model_joint.joint_model(num_img_channels=(6 + 2*config['use_flow']), output_exp=args.exploss, dropout_prob=config['dropout_prob']).to(device)
model.load_state_dict(torch.load(pretrained_path))

_, _, _, _, _, corr_traj, corr_traj_rot, est_traj, gt_traj, _, _, _ = test_trajectory(device, model, Reconstructor, test_dset_loaders, 0)

est_traj_se3 = [SE3.from_matrix(T, normalize=True) for T in est_traj]
corr_traj_rot_se3 = [SE3.from_matrix(T, normalize=True) for T in corr_traj_rot]
gt_traj_se3 = [SE3.from_matrix(T, normalize=True) for T in gt_traj]

dense_tm = TrajectoryMetrics(gt_traj_se3, gt_traj_se3,convention='Twv')
est_tm = TrajectoryMetrics(gt_traj_se3, est_traj_se3, convention='Twv')
corr_tm = TrajectoryMetrics(gt_traj_se3, corr_traj_rot_se3, convention = 'Twv')

tm_dict = {'Dense': dense_tm,
            'libviso2-s': est_tm,
               'Ours (Gen.)': corr_tm,
               }
est_vis = visualizers.TrajectoryVisualizer(tm_dict)
fig, ax = est_vis.plot_topdown(which_plane='xy', plot_gt=False, outfile = 'paper_plots_and_data/figs/{}.pdf'.format(seq[0]), title=r'{}'.format(seq[0]))

#for pretrained_path in glob.glob('{}**most_loop_closures**test_seq-{}**.pth'.format(output_dir, args.val_seq)):
#    print(pretrained_path)
#    test_dset = KittiLoaderPytorch(args.data_dir, config, [seq, seq, seq], mode='test', transform_img=get_data_transforms(config)['val'])
#    test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=4)
#    eval_dsets = {'test': test_dset_loaders}
#    Reconstructor = stn.Reconstructor().to(device)
#    model = mono_model_joint.joint_model(num_img_channels=(6 + 2*config['use_flow']), output_exp=args.exploss, dropout_prob=config['dropout_prob']).to(device)
#    model.load_state_dict(torch.load(pretrained_path))
#
#    for key, dset in eval_dsets.items():
#        ###plot images, depth map, explainability mask
#        img_idx = np.arange(999,1010,2)  #2627, 2632 for 00, 
#        img_array, disparity, exp_mask = test_depth_and_reconstruction(device, model, Reconstructor, dset, img_idx=img_idx)
#        img_array = img_array.view(-1,3,3,img_array.size(2),img_array.size(3))
#
#        for i in range(0,img_array.shape[0]):
#            plot_img_array(img_array[i,0], nrow=1, save_file = '{}{}{}.png'.format(figures_output_dir, '/imgs/', i))
#            plot_disp(disparity[i], save_file = '{}{}{}.png'.format(figures_output_dir, '/depth/', i))
#            if args.exploss:
#                plot_img_array(exp_mask[i], nrow=1, save_file = '{}{}{}.png'.format(figures_output_dir, '/exp_mask/', i))
#
#plot_img_array(img_array[1:,0], nrow=(len(img_idx)), save_file = '{}{}{}.png'.format(figures_output_dir, '/imgs/', 'img_full'))
#plot_img_array(exp_mask[:-1], nrow=(len(img_idx)), save_file = '{}{}{}.png'.format(figures_output_dir, '/imgs/', 'exp_mask_full'))