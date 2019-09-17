import time
import torch
import sys
sys.path.insert(0,'..')
from utils.learning_helpers import *
from utils.lie_algebra import se3_log_exp
import numpy as np
from liegroups import SE3
from pyslam.metrics import TrajectoryMetrics

def Validate(device, pose_model, spatial_trans, dset, loss):
    start = time.time()
    pose_model.train(False)  # Set model to evaluate mode
    pose_model.eval()        #used for batch normalization  # Set model to training mode
    spatial_trans.train(False)  
    spatial_trans.eval()        
    dset_size = dset.dataset.__len__()
    running_loss = 0.0           
        # Iterate over data.
    for data in dset:
            # get the inputs
        imgs, gt_lie_alg, intrinsics, vo_lie_alg, gt_correction = data
        gt_lie_alg = gt_lie_alg.type(torch.FloatTensor).to(device) 
        vo_lie_alg = vo_lie_alg.type(torch.FloatTensor).to(device)
        img_list = []
        for im in imgs: 
            img_list.append(im.to(device))

        intrinsics = intrinsics.type(torch.FloatTensor).to(device)[:,0,:,:] #only need one matrix since it's constant across the sequence
  
        corr, exp_mask, disparities = pose_model(img_list[0:3], vo_lie_alg)
        pose = se3_log_exp(corr, vo_lie_alg)
        minibatch_loss = loss.forward(img_list[-2], img_list[-1], pose, exp_mask, disparities, intrinsics, pose_vec_weight = vo_lie_alg, validate=True)

        running_loss += minibatch_loss.item()
     
    epoch_loss = running_loss / float(dset_size) 
    print('Validation Loss: {:.6f}'.format(epoch_loss))
    print("Validation epoch completed in {} seconds.".format(timeSince(start)))
    return epoch_loss

def test_depth_and_reconstruction(device, pose_model, spatial_trans,  dset, img_idx=[0,100,200,300]):
#    idx = np.random.randint(0,high=dset.dataset.__len__())
    exp_mask_array = torch.zeros(0)
    img_array = torch.zeros(0)
    disp_array = torch.zeros(0)
    for i in img_idx: #[1943,1944,1945,1946, 1947]:
        imgs, gt_lie_alg, intrinsics, vo_lie_alg, gt_correction = dset.dataset.__getitem__(i)
        gt_lie_alg = torch.FloatTensor(gt_lie_alg).to(device) 
        vo_lie_alg = torch.FloatTensor(vo_lie_alg).to(device)
        img_list = []
        for im in imgs:              
            img_list.append(im.to(device).unsqueeze(0))
        intrinsics = torch.FloatTensor(intrinsics).to(device)[0,:,:].unsqueeze(0)
    
        pose_model.train(False)  # Set model to evaluate mode
        pose_model.eval()        #used for batch normalization  # Set model to training mode
        spatial_trans.train(False)  
        spatial_trans.eval()        
        
        corr, exp_mask, disp = pose_model(img_list[0:3], vo_lie_alg.unsqueeze(0))
        ###comment for stereo
        exp_mask, disp = exp_mask[0], disp[0]
        disp = disp.unsqueeze(1)
        disp_array = torch.cat((disp_array, disp[0].cpu().detach()))
        depth = 1.0/disp[:,0].clone()
        pose = se3_log_exp(corr, vo_lie_alg)

        img_reconstructed = spatial_trans(img_list[-2], depth, -pose.clone(), intrinsics, intrinsics.inverse())    
        imgs = torch.stack((img_list[-2],img_reconstructed,img_list[-1]),dim=1)[0].cpu().detach()
        img_array = torch.cat((img_array, imgs))
        if exp_mask is not None:
            exp_mask = exp_mask.cpu().detach()
            exp_mask_array = torch.cat((exp_mask_array, exp_mask))

    return img_array, disp_array.numpy().squeeze(), exp_mask_array

def test_trajectory(device, pose_model, spatial_trans, dset, epoch):
    pose_model.train(False)  # Set model to evaluate mode
    pose_model.eval()        #used for batch normalization  # Set model to training mode
    spatial_trans.train(False)  
    spatial_trans.eval()     
    
    #initialize the relevant outputs
    full_corr_lie_alg_stacked, rot_corr_lie_alg_stacked, gt_lie_alg_stacked, vo_lie_alg_stacked, corrections_stacked, gt_corrections_stacked= \
            np.empty((0,6)), np.empty((0,6)), np.empty((0,6)), np.empty((0,6)), np.empty((0,6)), np.empty((0,6))

    for data in dset:
        imgs, gt_lie_alg, intrinsics, vo_lie_alg, gt_correction = data
        gt_lie_alg = gt_lie_alg.type(torch.FloatTensor).to(device)   
        vo_lie_alg = vo_lie_alg.type(torch.FloatTensor).to(device)
        img_list = []
        for im in imgs:              
            img_list.append(im.to(device))

        corr, exp_mask, disp = pose_model(img_list[0:3], vo_lie_alg)
        exp_mask, disp = exp_mask[0], disp[0][:,0]
        corr_rot = torch.clone(corr)
        corr_rot[:,0:3]=0

        corrected_pose = se3_log_exp(corr, vo_lie_alg)
        corrected_pose_rot_only = se3_log_exp(corr_rot, vo_lie_alg)
        
        
        corrections_stacked = np.vstack((corrections_stacked, corr.cpu().detach().numpy()))
        gt_corrections_stacked = np.vstack((gt_corrections_stacked, gt_correction.cpu().detach().numpy()))
        full_corr_lie_alg_stacked = np.vstack((full_corr_lie_alg_stacked, corrected_pose.cpu().detach().numpy()))
        rot_corr_lie_alg_stacked = np.vstack((rot_corr_lie_alg_stacked, corrected_pose_rot_only.cpu().detach().numpy()))
        gt_lie_alg_stacked = np.vstack((gt_lie_alg_stacked, gt_lie_alg.cpu().detach().numpy()))
        vo_lie_alg_stacked = np.vstack((vo_lie_alg_stacked, vo_lie_alg.cpu().detach().numpy()))

    est_traj, corr_traj, corr_traj_rot, gt_traj = [],[],[],[]
    gt_traj = dset.dataset.raw_gt_trials[0]
    est_traj.append(gt_traj[0])
    corr_traj.append(gt_traj[0])
    corr_traj_rot.append(gt_traj[0])

    cum_dist = [0]
    for i in range(0,full_corr_lie_alg_stacked.shape[0]):
        #classically estimated traj
        dT = SE3.exp(vo_lie_alg_stacked[i])
        new_est = SE3.as_matrix((dT.dot(SE3.from_matrix(est_traj[i],normalize=True).inv())).inv())
        est_traj.append(new_est)
        cum_dist.append(cum_dist[i]+np.linalg.norm(dT.trans))

        #corrected traj (rotation only)
        dT = SE3.exp(rot_corr_lie_alg_stacked[i])
        new_est = SE3.as_matrix((dT.dot(SE3.from_matrix(corr_traj_rot[i],normalize=True).inv())).inv())
        corr_traj_rot.append(new_est)
#        
#        
#        #corrected traj (full pose)
        dT = SE3.exp(full_corr_lie_alg_stacked[i])
        new_est = SE3.as_matrix((dT.dot(SE3.from_matrix(corr_traj[i],normalize=True).inv())).inv())
        corr_traj.append(new_est)

    gt_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in gt_traj]
    est_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in est_traj]
    corr_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in corr_traj]
    corr_traj_rot_se3 = [SE3.from_matrix(T,normalize=True) for T in corr_traj_rot]
    
    tm_est = TrajectoryMetrics(gt_traj_se3, est_traj_se3, convention = 'Twv')
    tm_corr = TrajectoryMetrics(gt_traj_se3, corr_traj_se3, convention = 'Twv')
    tm_corr_rot = TrajectoryMetrics(gt_traj_se3, corr_traj_rot_se3, convention = 'Twv')
    
    if epoch >= 0:
        est_mean_trans, est_mean_rot = tm_est.mean_err()
        corr_mean_trans, corr_mean_rot = tm_corr.mean_err()
        corr_rot_mean_trans, corr_rot_mean_rot = tm_corr_rot.mean_err()
        print("Odom. mean trans. error: {} | mean rot. error: {}".format(est_mean_trans, est_mean_rot*180/np.pi))
        print("Corr. mean trans. error: {} | mean rot. error: {}".format(corr_mean_trans, corr_mean_rot*180/np.pi))
        print("Corr. (rot. only) mean trans. error: {} | mean rot. error: {}".format(corr_rot_mean_trans, corr_rot_mean_rot*180/np.pi))
        
        seg_lengths = list(range(100,801,100))
        _, seg_errs_est = tm_est.segment_errors(seg_lengths, rot_unit='rad')
        _, seg_errs_corr = tm_corr.segment_errors(seg_lengths, rot_unit='rad')
        _, seg_errs_corr_rot = tm_corr_rot.segment_errors(seg_lengths, rot_unit='rad')
        print("Odom. mean Segment Errors: {} (trans, %) | {} (rot, deg/100m)".format(np.mean(seg_errs_est[:,1])*100, 100*np.mean(seg_errs_est[:,2])*180/np.pi))
        print("Corr. mean Segment Errors: {} (trans, %) | {} (rot, deg/100m)".format(np.mean(seg_errs_corr[:,1])*100, 100*np.mean(seg_errs_corr[:,2])*180/np.pi))
        print("Corr. (rot. only) mean Segment Errors: {} (trans, %) | {} (rot, deg/100m)".format(np.mean(seg_errs_corr_rot[:,1])*100, 100*np.mean(seg_errs_corr_rot[:,2])*180/np.pi)) 
        
    rot_seg_err = 100*np.mean(seg_errs_corr_rot[:,2])*180/np.pi

    return corrections_stacked, gt_corrections_stacked, full_corr_lie_alg_stacked, vo_lie_alg_stacked, gt_lie_alg_stacked, \
        np.array(corr_traj), np.array(corr_traj_rot), np.array(est_traj), np.array(gt_traj), rot_seg_err, corr_rot_mean_trans, np.array(cum_dist)
        
