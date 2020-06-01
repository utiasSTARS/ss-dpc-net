import torch
import sys
sys.path.insert(0,'..')
from data.kitti_loader import KittiLoaderPytorch
from train_mono import Train as Train_Mono
from validate import Validate, test_depth_and_reconstruction, test_trajectory
import models.stn as stn
import models.mono_model_joint as mono_model_joint
from utils.learning_helpers import *
from utils.custom_transforms import *
import losses
from vis import *
import numpy as np
import datetime
import time
from tensorboardX import SummaryWriter
import argparse
import scipy.io as sio
import torch.backends.cudnn as cudnn
import os
 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

parser = argparse.ArgumentParser(description='training arguments.')
parser.add_argument('--data_dir', type=str, default='/media/datasets/KITTI-dpc')
parser.add_argument('--date', type=str, default='0000000')
parser.add_argument('--train_seq', nargs='+', type=str, default=['all'])
parser.add_argument('--val_seq', nargs='+',type=str, default=['00'])
parser.add_argument('--test_seq', nargs='+', type=str, default=['05'])

parser.add_argument('--augment_motion', action='store_true', default=False) #skip samples to simulate faster motion
parser.add_argument('--augment_backwards', action='store_true', default=False) #reverse sequences to simulate backwards motion
parser.add_argument('--img_per_sample', type=int, default=2) ## 1 target image, and  'img_per_sample -1 source images.
parser.add_argument('--correction_rate', type=int, default=1) ## if not one, only perform corrections every 'correction_rate' frames (samples become {1,3},{3,5},{5,7} when rate is 2)
parser.add_argument('--skip', type=int, default=1) ## if not one, skip every 'skip' samples that are generated ({1,2}, {2,3}, {3,4} becomes {1,2}, {3,4}) 
parser.add_argument('--exploss', action='store_true', default=True) ## For using the explainability loss 
parser.add_argument('--exp_weight', type=float, default=0.18)
parser.add_argument('--use_flow', action='store_true',default=True)  ## For using optical flow as a network input
parser.add_argument('--normalize_img', action='store_true',default=True) ## Normalize images for network
parser.add_argument("--estimator_type", type=str, default='mono') ## mono or stereo (the type of VO estimator being corrected)

parser.add_argument('--minibatch', type=int, default=32) ## minibatch size
parser.add_argument('--wd', type=float, default=0)  ## weight decay
parser.add_argument('--lr', type=float, default=9e-4) ## Learning rate
parser.add_argument('--num_epochs', type=int, default=20) ## Number of training epochs
parser.add_argument('--lr_decay_epoch', type=float, default=4) ## decay learning rate every X epochs
parser.add_argument('--dropout_prob', type=float, default=0.3) ## dropout for posenet
parser.add_argument('--save_results', action='store_true', default=False) 

args = parser.parse_args()
config={
    'num_frames': None, ## if used, only take num_frames from each sequence for training
    }
for k in args.__dict__:
    config[k] = args.__dict__[k]

dsets = {x: KittiLoaderPytorch(args.data_dir, config, [args.train_seq, args.val_seq, args.test_seq], mode=x, transform_img=get_data_transforms(config)[x], num_frames = config['num_frames'], \
                               augment=config['augment_motion'], skip=config['skip'], augment_backwards=config['augment_backwards']) for x in ['train', 'val']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=config['minibatch'], shuffle=True, num_workers=4) for x in ['train', 'val']}

val_dset = KittiLoaderPytorch(args.data_dir, config, [args.train_seq, args.val_seq, args.test_seq], mode='val', transform_img=get_data_transforms(config)['val'])
val_dset_loaders = torch.utils.data.DataLoader(val_dset, batch_size=config['minibatch'], shuffle=False, num_workers=4)

test_dset = KittiLoaderPytorch(args.data_dir, config, [args.train_seq, args.val_seq, args.test_seq], mode='test', transform_img=get_data_transforms(config)['val'])
test_dset_loaders = torch.utils.data.DataLoader(test_dset, batch_size=config['minibatch'], shuffle=False, num_workers=4)

eval_dsets = {'val': val_dset_loaders, 'test':test_dset_loaders}

def main():
    results = {}
    results['estimator'] = config['estimator_type']
    start = time.time()
    now= datetime.datetime.now()
    ts = '{}-{}-{}-{}-{}'.format(now.year, now.month, now.day, now.hour, now.minute)

    criterion = losses.photometric_reconstruction_loss()
    exp_loss = losses.explainability_loss()
    Reconstructor = stn.Reconstructor().to(device)
    loss = losses.Compute_Loss(Reconstructor, criterion, exp_loss, exp_weight = config['exp_weight'])
    model = mono_model_joint.joint_model(num_img_channels=(6 + 2*config['use_flow']), output_exp=args.exploss, dropout_prob=config['dropout_prob']).to(device)

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=config['lr'], weight_decay = config['wd'])

    cudnn.benchmark = True
    
    est_traj_stacked ={}
    corr_pose_change_vecs_stacked = {}
    corr_stacked = {}
    losses_stacked = {}
    best_val_loss, best_rot_seg_err, best_trans_err, most_loop_closure = {}, {}, {}, {}
    best_rot_acc_epoch, best_trans_acc_epoch, best_loss_epoch, most_loop_closure_epoch = {}, {}, {}, {}
    for key, dset in eval_dsets.items():
        est_traj_stacked[key] = np.empty((0,eval_dsets[key].dataset.raw_gt_trials[0].shape[0], 4, 4))
        losses_stacked[key] = np.empty((0, eval_dsets[key].dataset.raw_gt_trials[0].shape[0]-1))
        corr_pose_change_vecs_stacked[key] = np.empty((0, eval_dsets[key].dataset.raw_gt_trials[0].shape[0]-1,6))
        corr_stacked[key] = np.copy(corr_pose_change_vecs_stacked[key])
        best_val_loss[key], best_rot_seg_err[key], best_trans_err[key], most_loop_closure[key]  = 1e5, 1e5, 1e5, 0

    
    for epoch in range(0,config['num_epochs']):
        optimizer = exp_lr_scheduler(model, optimizer, epoch, lr_decay_epoch=config['lr_decay_epoch']) ## reduce learning rate as training progresses  
        
        train_loss = Train_Mono(device,model, Reconstructor, dset_loaders['train'], loss, optimizer, epoch)
        val_loss = Validate(device, model, Reconstructor, dset_loaders['val'], loss)      
#        
        if epoch == 0:
            writer = SummaryWriter(comment="-val_seq-{}-test_seq-{}".format(args.val_seq[0], args.test_seq[0]))
        writer.add_scalars('',{'train': train_loss, 'val': val_loss}, epoch+1)
        for key, dset in eval_dsets.items():
            print("{} Set, Epoch {}".format(key, epoch))
            
            ###plot images, depth map, explainability mask
            img_array, disparity, exp_mask = test_depth_and_reconstruction(device, model, Reconstructor, dset)
            img_array = plot_img_array(img_array)
            depth = plot_disp(disparity[-1])
            writer.add_image(key+'/imgs',img_array,epoch+1) 
            writer.add_image(key+'/depth', depth, epoch+1)
            
            if args.exploss:
                exp_mask = plot_img_array(exp_mask)
                writer.add_image(key+'/exp_mask', exp_mask, epoch+1)
                
            ###plot trajectories    
            corr, gt_corr, corr_pose_change_vec, odom_pose_change_vec, gt_pose_change_vec, corr_traj, corr_traj_rot, est_traj, gt_traj, \
                rot_seg_err, trans_err, cum_dist = test_trajectory(device, model, Reconstructor, dset, epoch)

            corrections = plot_6_by_1(corr, title = 'Corrections')                                          
            correction_errors = plot_6_by_1(np.abs(corr - gt_corr), title='6x1 Errors')    
            est_traj_img = plot_multi_traj(est_traj, 'Odom.', gt_traj, 'GT', key+' Set')
            corr_traj_img = plot_multi_traj(corr_traj, 'corr.', gt_traj, 'GT', key+' Set')
            corr_traj_rot_img = plot_multi_traj(corr_traj_rot, 'corr.', gt_traj, 'GT', key+' Set')

            writer.add_image(key+'/corr_traj', corr_traj_img, epoch+1)
            writer.add_image(key+'/corr_traj (Rot. only)', corr_traj_rot_img, epoch+1)
            writer.add_image(key+'/est_traj', est_traj_img, epoch+1)
            writer.add_image(key+'/correction_errors', correction_errors, epoch+1)
   
            corr_stacked[key] = np.vstack((corr_stacked[key], corr.reshape((1,-1,6))))
            est_traj_stacked[key] = np.vstack((est_traj_stacked[key], corr_traj_rot.reshape((1,-1,4,4))))
            corr_pose_change_vecs_stacked[key] = np.vstack((corr_pose_change_vecs_stacked[key], corr_pose_change_vec.reshape((1,-1,6))))

            results[key] = {'val_seq': args.val_seq, 
                   'test_seq': args.test_seq,
               'epochs': epoch+1,
               'est_traj': est_traj_stacked[key],
               'corrections': corr_stacked[key],
               'est_traj_reconstruction_loss': losses_stacked[key],
               'corr_pose_vecs': corr_pose_change_vecs_stacked[key],
               'odom_pose_vecs': odom_pose_change_vec,
               'gt_traj': gt_traj, 
               }
             
            if args.save_results:   
                os.makedirs('results/{}'.format(config['date']), exist_ok=True)
                if config['estimator_type'] == 'mono':
                    _, num_loop_closure, _ = find_loop_closures(corr_traj, cum_dist)
                if config['estimator_type'] == 'stereo':
                    _, num_loop_closure, _ = find_loop_closures(corr_traj_rot, cum_dist)
                print("{} Loop Closures detected".format(num_loop_closure))
                ##Save the best models
                
                if val_loss < best_val_loss[key]:
                    best_val_loss[key] = val_loss
                    best_loss_epoch[key] = epoch
                    state_dict_loss = model.state_dict()
                    print("Lowest validation loss (saving model)")       
                    if key == 'val':
                        torch.save(state_dict_loss, 'results/{}/{}-best-loss-val_seq-{}-test_seq-{}.pth'.format(config['date'], ts, args.val_seq[0], args.test_seq[0]))
                if rot_seg_err < best_rot_seg_err[key]:
                    best_rot_seg_err[key] = rot_seg_err
                    best_rot_acc_epoch[key] = epoch
                    state_dict_acc = model.state_dict()
                    print("Lowest error (saving model)")
                    if key == 'val':
                        torch.save(state_dict_acc, 'results/{}/{}-best_rot_acc-val_seq-{}-test_seq-{}.pth'.format(config['date'], ts, args.val_seq[0], args.test_seq[0]))
                if trans_err < best_trans_err[key]:
                    best_trans_err[key] = trans_err
                    best_trans_acc_epoch[key] = epoch
                    state_dict_trans = model.state_dict()
                    print("Lowest position error (saving model)")
                    if key == 'val':
                        torch.save(state_dict_trans, 'results/{}/{}-best_trans_acc-val_seq-{}-test_seq-{}.pth'.format(config['date'], ts, args.val_seq[0], args.test_seq[0]))
                       
                if num_loop_closure >= most_loop_closure[key]:
                    most_loop_closure[key] = num_loop_closure
                    most_loop_closure_epoch[key] = epoch
                    state_dict_loop_closure = model.state_dict()
                    print("Most Loop Closures detected ({})".format(most_loop_closure[key]))
                    if key == 'val':
                        torch.save(state_dict_loop_closure, 'results/{}/{}-most_loop_closures-val_seq-{}-test_seq-{}.pth'.format(config['date'], ts, args.val_seq[0], args.test_seq[0]))
                results[key]['best_rot_acc_epoch'] = best_rot_acc_epoch[key]
                results[key]['best_trans_acc_epoch'] = best_trans_acc_epoch[key]
                results[key]['best_loss_epoch'] = best_loss_epoch[key]
                results[key]['best_loop_closure_epoch'] = most_loop_closure_epoch[key] 
                sio.savemat('results/{}/{}-results-val_seq-{}-test_seq-{}.mat'.format(config['date'], ts, args.val_seq[0], args.test_seq[0]),results)
                f = open("results/{}/{}-config.txt".format(config['date'],ts),"w")
                f.write( str(config) )
                f.close()

    duration = timeSince(start)    
    print("Training complete (duration: {})".format(duration))
 
main()
