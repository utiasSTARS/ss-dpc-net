import pykitti
import numpy as np
import scipy.io as sio
import imageio
import os
import concurrent.futures
from PIL import Image
import argparse
from liegroups import SE3
import glob

parser = argparse.ArgumentParser(description='rguments.')
parser.add_argument("--source_dir", type=str, default='/media/brandon/DATA/KITTI-odometry/dataset/')
parser.add_argument("--target_dir", type=str, default='/media/brandon/DATA/KITTI-odometry-dpc/')
parser.add_argument("--remove_static", action='store_true', default=True)
args = parser.parse_args()

target_dir = args.target_dir
os.makedirs(target_dir, exist_ok=True)
seq_info = {}
sequences = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

args.height = 240
args.width = 376

stereo_libviso2_dir = 'libviso2-estimates/stereo-odometry/'

def load_image(img_file):
    img_height = args.height 
    img_width = args.width 
    img = np.array(Image.open(img_file))
    orig_img_height = img.shape[0]
    orig_img_width = img.shape[1]
    zoom_y = img_height/orig_img_height
    zoom_x = img_width/orig_img_width
    img = np.array(Image.fromarray(img).resize((img_width, img_height)))
    return img, zoom_x, zoom_y, orig_img_width, orig_img_height

    ###Iterate through all specified KITTI sequences and extract raw data, and trajectories
frames = None
for seq in sequences:
    data = pykitti.odometry(args.source_dir, seq, frames=frames)
        ###make the new directories
    seq_dir = os.path.join(target_dir, seq)
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(os.path.join(seq_dir, 'image_2'), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, 'image_3'), exist_ok=True)
 
        ###store filenames of camera data, and intrinsic matrix
    seq_info['intrinsics'] = np.array(data.calib.K_cam2).reshape((-1,3,3)).repeat(len(data.cam2_files),0)
    i = 0
    with concurrent.futures.ProcessPoolExecutor() as executor: 
        for filename, output in zip(data.cam2_files, executor.map(load_image, data.cam2_files)):
            img, zoomx, zoomy, orig_img_width, orig_img_height = output
            new_filename = os.path.join(target_dir, filename.split(args.source_dir)[1]).replace('sequences/','').replace('.png','.jpg')
            imageio.imwrite(new_filename, img)
            seq_info['intrinsics'][i,0] *= zoomx
            seq_info['intrinsics'][i,1] *= zoomy
            data.cam2_files[i] = np.array(new_filename)
            i+=1
        i = 0
        for filename, output in zip(data.cam3_files, executor.map(load_image, data.cam3_files)):
            img, zoomx, zoomy, orig_img_width, orig_img_height = output
            new_filename = os.path.join(target_dir, filename.split(args.source_dir)[1]).replace('sequences/','').replace('.png','.jpg')
            data.cam3_files[i] = np.array(new_filename)
            i+=1

    seq_info['cam_02'] = np.array(data.cam2_files)
    seq_info['cam_03'] = np.array(data.cam3_files)
     
        ###Import libviso2 estimate for correcting
    stereo_libviso2_data = sio.loadmat(stereo_libviso2_dir+seq+'.mat')  #sparse VO
    stereo_traj = stereo_libviso2_data['poses_est'].transpose(2,0,1)
        ### store the ground truth pose
    seq_info['sparse_gt_pose'] = stereo_libviso2_data['poses_gt'].transpose(2,0,1)

    stereo_seq_info = seq_info.copy()
    
    ### store the VO pose estimates to extract 
    stereo_seq_info['sparse_vo'] = stereo_traj
    
        ###filter out frames with low rotational or translational velocities
    for seq_info in [stereo_seq_info]:
        if args.remove_static:
            print("Removing Static frames from {}".format(seq))            
            deleting = True
            
            while deleting:
                idx_list = []
                sparse_traj = np.copy(seq_info['sparse_vo'])
                for i in range(0,sparse_traj.shape[0]-1,2):
                    T2 = SE3.from_matrix(sparse_traj[i+1,:,:], normalize=True).inv()
                    T1 = SE3.from_matrix(sparse_traj[i,:,:], normalize=True)
                    dT = T2.dot(T1)
                    pose_vec = dT.log()
                    trans_norm = np.linalg.norm(pose_vec[0:3])
                    rot_norm = np.linalg.norm(pose_vec[3:6])
                    if trans_norm < 1.5 and rot_norm < 0.007: #0.007
                        idx_list.append(i)
#                            os.remove(seq_info['cam_02'][i])
#                            os.remove(seq_info['cam_03'][i])
                if len(idx_list) == 0:
                    deleting = False
                
                print('deleting {} frames'.format(len(idx_list)))
                print('original length: {}'.format(seq_info['cam_02'].shape))
                
                seq_info['intrinsics'] = np.delete(seq_info['intrinsics'],idx_list,axis=0)
                seq_info['cam_02'] = np.delete(seq_info['cam_02'],idx_list,axis=0)
                seq_info['cam_03'] = np.delete(seq_info['cam_03'],idx_list,axis=0)
                seq_info['sparse_gt_pose'] = np.delete(seq_info['sparse_gt_pose'],idx_list,axis=0)
                seq_info['sparse_vo'] = np.delete(seq_info['sparse_vo'],idx_list,axis=0)
                print('final length: {}'.format(seq_info['cam_02'].shape))
 
    sio.savemat(seq_dir + '/stereo_data.mat', stereo_seq_info)