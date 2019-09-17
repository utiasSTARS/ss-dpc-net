import time
import torch
import sys
sys.path.insert(0,'..')
from utils.learning_helpers import *
from utils.lie_algebra import se3_log_exp

def Train(device, pose_model, spatial_trans, dset, loss, optimizer,epoch):
    start = time.time()
    pose_model.train(True)  # Set model to training mode
    spatial_trans.train(False)
    dset_size = dset.dataset.__len__()
    print("train dset size", dset_size)
    running_loss = 0.0           
        # Iterate over data.
    for data in dset:
            # get the inputs (we only use the images, intrinsics, and vo_lie_alg)
        imgs, _, intrinsics, vo_lie_alg, _ = data
        vo_lie_alg = vo_lie_alg.type(torch.FloatTensor).to(device)
        
        img_list = []
        for im in imgs: 
            img_list.append(im.to(device))

        intrinsics = intrinsics.type(torch.FloatTensor).to(device)[:,0,:,:] #only need one matrix since it's constant across the sequence

        corr, exp_mask, disparities = pose_model(img_list[0:3], vo_lie_alg)
        pose = se3_log_exp(corr, vo_lie_alg)

        minibatch_loss = loss(img_list[-2], img_list[-1], pose, exp_mask, disparities, intrinsics, pose_vec_weight = vo_lie_alg)

        optimizer.zero_grad()
        minibatch_loss.backward()
#        torch.nn.utils.clip_grad_norm_(pose_model.parameters(), clip)
        optimizer.step()
        running_loss += minibatch_loss.item()
        
    epoch_loss = running_loss / float(dset_size)

    print('Training Loss: {:.6f}'.format(epoch_loss))
    print("Training epoch completed in {} seconds.".format(timeSince(start)))
    return epoch_loss


