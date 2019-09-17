import scipy.io as sio
import numpy as np
from liegroups import SE3
from pyslam.metrics import TrajectoryMetrics
import csv
import glob
stats_list = []
data_dir = 'best_mono'
dense_results = {'00': ['', 'Dense', '---', 12.41, 2.45, 1.28, 0.542],
                 '02': ['', 'Dense', '---', 16.33, 3.19, 1.21, 0.467],
                 '05': ['', 'Dense', '---', 5.83, 2.05, 0.69, 0.320],
                 }

dpcnet_results = {'00': ['', 'DPC-Net', '---', 15.68, 3.07, 1.62, 0.559 ],
                 '02': ['', 'DPC-Net', '---', 17.69, 2.86, 1.16, 0.436],
                 '05': ['', 'DPC-Net', '---', 9.82, 3.57, 1.34, 0.562],
                 }

def generate_trajectory_metrics(gt_traj, est_traj, name='',seq='', mode=''):
    gt_traj_se3 = [SE3.from_matrix(T,normalize=True) for T in gt_traj]
    est_traj_se3 = [SE3.from_matrix(T, normalize=True) for T in est_traj]
    tm = TrajectoryMetrics(gt_traj_se3, est_traj_se3, convention = 'Twv')
    
    est_mATE_trans, est_mATE_rot = tm.mean_err()
    est_mATE_rot = est_mATE_rot*180/np.pi
    print("{} mean trans. error: {} | mean rot. error: {}".format(name, est_mATE_trans, est_mATE_rot))
    
    seg_lengths = list(range(100,801,100))
    _, est_seg_errs = tm.segment_errors(seg_lengths, rot_unit='rad')
    est_seg_err_trans = np.mean(est_seg_errs[:,1])*100
    est_seg_err_rot = 100*np.mean(est_seg_errs[:,2])*180/np.pi
    print("{} mean Segment Errors: {} (trans, %) | {} (rot, deg/100m)".format(name, est_seg_err_trans, est_seg_err_rot) )
    return tm, (seq, name, mode, est_mATE_trans.round(3), est_mATE_rot.round(3), est_seg_err_trans.round(3), est_seg_err_rot.round(3))

    #Iterate through all models within a directory
for f in sorted(glob.iglob('{}/**.mat'.format(data_dir), recursive=True)):
    print(f)
    matfile = sio.loadmat(f)
    estimator = matfile['estimator']
    mode = matfile['mode']
    print(estimator, mode)
    matfile_val = matfile['val']
    matfile = matfile['test']
    seq = matfile['test_seq'].item().item()
    print(seq)
    gt_traj = matfile['gt_traj'].item()    #ground truth pose (seq_size x 4 x 4)

    best_loss_epoch = matfile_val['best_loss_epoch'].item().item()
    best_loop_closure_epoch = matfile_val['best_loop_closure_epoch'].item().item()
    odom_pose_vec = matfile['odom_pose_vecs'].item() #original odometry 6x1 vecs (seq_size-1 x 6)
    corr_pose_vec = matfile['corr_pose_vecs'].item() # corrected 6x1 vecs (seq_size-1 x 6)
    
    if estimator == 'stereo': #use rotation corrections only.
        corr_pose_vec[:,:,0:3] = odom_pose_vec[:,0:3]
        
    ###Use the epoch with the lowest loss, or most loop closures:
    best_loss_pose_vec = corr_pose_vec[best_loss_epoch]
    best_loop_closure_pose_vec = corr_pose_vec[best_loop_closure_epoch]

    est_traj, best_loss_traj, best_loop_closure_traj = [],[],[]
    est_traj.append(gt_traj[0])
    best_loss_traj.append(gt_traj[0])
    best_loop_closure_traj.append(gt_traj[0])
    
    for i in range(0,odom_pose_vec.shape[0]):
        #classically estimated traj
        dT = SE3.exp(odom_pose_vec[i])
        new_est = SE3.as_matrix((dT.dot(SE3.from_matrix(est_traj[i], normalize=True).inv())).inv())
        est_traj.append(new_est)
        
        #best validation loss traj
        dT_corr = SE3.exp(best_loss_pose_vec[i])
        new_est = SE3.as_matrix((dT_corr.dot(SE3.from_matrix(best_loss_traj[i], normalize=True).inv())).inv())
        best_loss_traj.append(new_est)         

        #best loop closure traj
        dT_corr = SE3.exp(best_loop_closure_pose_vec[i])
        new_est = SE3.as_matrix((dT_corr.dot(SE3.from_matrix(best_loop_closure_traj[i], normalize=True).inv())).inv())
        best_loop_closure_traj.append(new_est)  

    est_tm, est_metrics =  generate_trajectory_metrics(gt_traj, est_traj, name='libviso2', seq=seq, mode='---')
    best_loss_tm, best_loss_metrics = generate_trajectory_metrics(gt_traj, best_loss_traj, name='Ours', seq='', mode='best loss')
    best_loop_closure_tm, best_loop_closure_metrics = generate_trajectory_metrics(gt_traj, best_loop_closure_traj, name='', seq='', mode='loop closure')
      
    stats_list.append(est_metrics)
#    stats_list.append(dense_results[str(seq)])
#    stats_list.append(dpcnet_results[str(seq)])
    stats_list.append(best_loss_metrics)
    stats_list.append(best_loop_closure_metrics)
    stats_list.append('')
    
csv_filename = 'mono_results_kitti.csv'
csv_header1 = ['', '', '', 'm-ATE', '', 'Mean Segment Errors', '']
csv_header2 = ['Sequence (Length)', 'Estimator', 'Mode', 'Trans. (m)', 'Rot. (deg)', 'Trans. (%)', 'Rot. (deg/100m)']
with open(csv_filename, "w") as f:
    writer = csv.writer(f)
    writer.writerow(csv_header1)
    writer.writerow(csv_header2)
    writer.writerows(stats_list)
        
