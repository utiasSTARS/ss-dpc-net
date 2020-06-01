import scipy.io as sio
import numpy as np
from liegroups import SO3, SE3
from pyslam.metrics import TrajectoryMetrics
from generate_stereo_results import generate_trajectory_metrics
import visualizers
import csv
import glob
from collections import OrderedDict
stats_list = []

dense_results = {'00': ['', 'Dense', 12.41, 2.45, 1.28, 5.42],
                 '02': ['', 'Dense', 16.33, 3.19, 1.21, 4.67],
                 '05': ['', 'Dense', 5.83, 2.05, 0.69, 3.20],
                 }

data_dir = 'best_mono'

for f in sorted(glob.iglob('{}/**.mat'.format(data_dir), recursive=True)):
    print(f)
    matfile = sio.loadmat(f)
    estimator = matfile['estimator']

    matfile_val = matfile['val']
    matfile = matfile['test']
    seq = matfile['test_seq'].item().item()
    best_loop_closure_epoch = matfile_val['best_loop_closure_epoch'].item().item()
    gt_traj = matfile['gt_traj'].item()
    odom_pose_vec = matfile['odom_pose_vecs'].item() #original odometry 6x1 vecs (seq_size-1 x 6)
    corr_pose_vec = matfile['corr_pose_vecs'].item() # corrected 6x1 vecs (seq_size-1 x 6)
    if estimator == 'stereo': #use rotation corrections only.
        corr_pose_vec[:,:,0:3] = odom_pose_vec[:,0:3]
    best_loop_closure_pose_vec = corr_pose_vec[best_loop_closure_epoch]

    
    est_traj, avg_corr_traj, dense_traj, dense_gt = [],[],[],[]
    est_traj.append(gt_traj[0])
    avg_corr_traj.append(gt_traj[0])
  
    for i in range(0,odom_pose_vec.shape[0]):
        #classically estimated traj
        dT = SE3.exp(odom_pose_vec[i])
        new_est = SE3.as_matrix((dT.dot(SE3.from_matrix(est_traj[i], normalize=True).inv())).inv())
        est_traj.append(new_est)
        
        dT_corr = SE3.exp(best_loop_closure_pose_vec[i])
        new_est = SE3.as_matrix((dT_corr.dot(SE3.from_matrix(avg_corr_traj[i], normalize=True).inv())).inv())
        avg_corr_traj.append(new_est)    
 
    est_tm, est_metrics =  generate_trajectory_metrics(gt_traj, est_traj, name='libviso2', seq=seq)
    corr_tm, avg_corr_metrics = generate_trajectory_metrics(gt_traj, avg_corr_traj, name='ss-dpcnet', seq='')
    saved_traj[seq]['ours'] = corr_tm
    saved_traj[seq]['libviso2'] = est_tm


for seq in ['09', '10']:
    tm_dict = {'Libviso2-m': saved_traj[seq]['libviso2'],
               'Ours (Corrected)': saved_traj[seq]['ours'],
               }
    order_of_keys = ["Libviso2-m", "Ours (Corrected)"]
    list_of_tuples = [(key, tm_dict[key]) for key in order_of_keys]
    tm_dict = OrderedDict(list_of_tuples)
    
    est_vis = visualizers.TrajectoryVisualizer(tm_dict)
    plt.figure()
    fig, ax = est_vis.plot_topdown(which_plane='xy', outfile = 'figs/mono-seq-{}.pdf'.format(seq), title=r'{}'.format(seq))
