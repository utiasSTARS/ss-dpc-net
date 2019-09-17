import math
import torch
import time
from torch.optim import Optimizer
import numpy as np
from liegroups import SE3, SO3

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def save_state(state, filename='test.pth.tar'):
    torch.save(state, filename)
    
def exp_lr_scheduler(model, optimizer, epoch, lr_decay_epoch=5):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch == 70 or epoch == 71 or epoch == 72 or epoch == 73 or epoch == 74 or epoch == 75:
        print('LR is reduced by {}'.format(0.5))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            param_group['lr'] = param_group['lr']*0.5
            
    if epoch !=0 and epoch%lr_decay_epoch==0:
#        lr_decay_factor = (0.5**(epoch // lr_decay_epoch))

#    if epoch % lr_decay_epoch == 0:
        print('LR is reduced by {}'.format(0.5))
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            param_group['lr'] = param_group['lr']*0.5

    return optimizer   

def find_loop_closures(traj, cum_dist):
    num_loop_closures = 0
    filtered_loop_closures=0
    idx_list=[]
    for i in range(0,traj.shape[0],8): #check for loop closure points (compare current frame with all future points, and do this for every 20th frame)
        current_pose = traj[i]
        current_trans = current_pose[0:3,3]
        current_rot = SO3.from_matrix(current_pose[0:3,0:3], normalize=True).to_rpy()
        current_yaw = current_rot[2]
        
        current_cum_dist = cum_dist[i]
        loop_closure_idx = np.linalg.norm(np.abs(current_trans[0:3] - traj[i+1:,0:3,3]),axis=1) <= 7
        dist_idx = (cum_dist[i+1:]-current_cum_dist) >=10
        loop_closure_idx = loop_closure_idx & dist_idx
        
        idx = np.where(loop_closure_idx == 1)
        
        if idx != np.array([]):
            for pose_idx in idx[0]:
                T = traj[i+1:][pose_idx]
                yaw = SE3.from_matrix(T,normalize=True).rot.to_rpy()[2]
                yaw_diff = np.abs(np.abs(current_yaw) - np.abs(yaw))
                in_range = ((yaw_diff <= 0.15)) or (np.abs((np.pi - yaw_diff) <=0.15))
                filtered_loop_closures += in_range
                if in_range:
                    idx_list.append(pose_idx+i)
        
        num_loop_closures += np.sum(loop_closure_idx)

    return num_loop_closures, filtered_loop_closures, idx_list