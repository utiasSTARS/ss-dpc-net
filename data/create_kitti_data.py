import pykitti
import numpy as np
import scipy.io as sio
import imageio
import os
import concurrent.futures
from PIL import Image
import argparse
from liegroups import SE3

parser = argparse.ArgumentParser(description='arguments.')
parser.add_argument("--source_dir", type=str, default='/media/datasets/KITTI/raw/')
parser.add_argument("--target_dir", type=str, default='/media/datasets/KITTI-dpc/')
parser.add_argument("--remove_static", action='store_true', default=True)
args = parser.parse_args()

target_dir = args.target_dir
os.makedirs(target_dir, exist_ok=True)
seq_info = {}
seq = ['00', '02', '05', '06', '07', '08', '09', '10', '09_26_0001', '09_26_0005', '09_26_0009', '09_26_0014', '09_26_0018', '09_26_0019', \
        '09_26_0022', '09_26_0023', '09_26_0029', '09_26_0035', '09_26_0036', '09_26_0046', '09_26_0051', '09_26_0061', '09_26_0064', '09_26_0070', \
        '09_26_0079', '09_26_0084', '09_26_0086', '09_26_0087', '09_26_0091', '09_26_0095', '09_26_0096', '09_26_0117']

args.height = 240
args.width = 376

mono_libviso2_dir = 'libviso2-estimates/mono/'
stereo_libviso2_dir = 'libviso2-estimates/stereo/'
global KITTI_SEQS_DICT
KITTI_SEQS_DICT = {'00': {'date': '2011_10_03',
            'drive': '0027',
            'frames': range(0, 4541)},
        '01': {'date': '2011_10_03',
            'drive': '0042',
            'frames': range(0, 1101)},
        '02': {'date': '2011_10_03',
            'drive': '0034',
            'frames': range(0, 4661)},
        '04': {'date': '2011_09_30',
            'drive': '0016',
            'frames': range(0, 271)},
        '05': {'date': '2011_09_30',
            'drive': '0018',
            'frames': range(0, 2761)},
        '06': {'date': '2011_09_30',
            'drive': '0020',
            'frames': range(0, 1101)},
        '07': {'date': '2011_09_30',
            'drive': '0027',
            'frames': range(0, 1101)},
        '08': {'date': '2011_09_30',
            'drive': '0028',
            'frames': range(1100, 5171)},
        '09': {'date': '2011_09_30',
            'drive': '0033',
            'frames': range(0, 1591)},
        '10': {'date': '2011_09_30',
            'drive': '0034',
            'frames': range(0, 1201)},
        '09_26_0001': {'date': '2011_09_26',
            'drive': '0001',
            'frames': None},
        '09_26_0005': {'date': '2011_09_26',
            'drive': '0005',
            'frames': None},
        '09_26_0009': {'date': '2011_09_26',
            'drive': '0009',
            'frames': None},
        '09_26_0014': {'date': '2011_09_26',
            'drive': '0014',
            'frames': None},
        '09_26_0018': {'date': '2011_09_26',
            'drive': '0018',
            'frames': None},
        '09_26_0019': {'date': '2011_09_26',
            'drive': '0019',
            'frames': None},
        '09_26_0022': {'date': '2011_09_26',
            'drive': '0022',
            'frames': None},
        '09_26_0023': {'date': '2011_09_26',
            'drive': '0023',
            'frames': None},
        '09_26_0029': {'date': '2011_09_26',
            'drive': '0029',
            'frames': None},                       
        '09_26_0035': {'date': '2011_09_26',
            'drive': '0035',
            'frames': None},
        '09_26_0036': {'date': '2011_09_26',
            'drive': '0036',
            'frames': None},
        '09_26_0046': {'date': '2011_09_26',
            'drive': '0046',
            'frames': None},
        '09_26_0051': {'date': '2011_09_26',
            'drive': '0051',
            'frames': None},
        '09_26_0061': {'date': '2011_09_26',
            'drive': '0061',
            'frames': None},
        '09_26_0064': {'date': '2011_09_26',
            'drive': '0064',
            'frames': None},
        '09_26_0070': {'date': '2011_09_26',
            'drive': '0070',
            'frames': None},
        '09_26_0079': {'date': '2011_09_26',
            'drive': '0079',
            'frames': None},
        '09_26_0084': {'date': '2011_09_26',
            'drive': '0084',
            'frames': None},
        '09_26_0086': {'date': '2011_09_26',
            'drive': '0086',
            'frames': None},
        '09_26_0087': {'date': '2011_09_26',
            'drive': '0087',
            'frames': None},
        '09_26_0091': {'date': '2011_09_26',
            'drive': '0091',
            'frames': None},
        '09_26_0095': {'date': '2011_09_26',
            'drive': '0095',
            'frames': None},
        '09_26_0096': {'date': '2011_09_26',
            'drive': '0096',
            'frames': None},
        '09_26_0117': {'date': '2011_09_26',
            'drive': '0117',
            'frames': None},
                        
        }
        
def load_image(img_file):
    img_height = args.height #240 #360 #
    img_width = args.width #376 #564 
    img = np.array(Image.open(img_file))
    orig_img_height = img.shape[0]
    orig_img_width = img.shape[1]
    zoom_y = img_height/orig_img_height
    zoom_x = img_width/orig_img_width
#    img = np.array(Image.fromarray(img).crop([425, 65, 801, 305]))
    img = np.array(Image.fromarray(img).resize((img_width, img_height)))
    return img, zoom_x, zoom_y, orig_img_width, orig_img_height

    ###Iterate through all specified KITTI sequences and extract raw data, and trajectories
for s in range(0,len(seq)):     
    date = KITTI_SEQS_DICT[seq[s]]['date']
    drive = KITTI_SEQS_DICT[seq[s]]['drive']
    print(drive)
    frames = KITTI_SEQS_DICT[seq[s]]['frames']

    data = pykitti.raw(args.source_dir, date, drive, frames=frames)
        ###make the new directories
    seq_dir = os.path.join(target_dir, date+'_drive_'+drive+'_sync')
    os.makedirs(seq_dir, exist_ok=True)
    os.makedirs(os.path.join(seq_dir, 'image_02'), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, 'image_03'), exist_ok=True)
    
        ###store filenames of camera data, and intrinsic matrix
    seq_info['intrinsics'] = np.array(data.calib.K_cam2).reshape((-1,3,3)).repeat(len(data.oxts),0)
    i = 0
    with concurrent.futures.ProcessPoolExecutor() as executor: 
        for filename, output in zip(data.cam2_files, executor.map(load_image, data.cam2_files)):
            img, zoomx, zoomy, orig_img_width, orig_img_height = output
            new_filename = os.path.join(target_dir, filename.split(args.source_dir)[1]).replace('data/','').replace('.png','.jpg').replace('/'+date+'/','/')
            imageio.imwrite(new_filename, img)
            seq_info['intrinsics'][i,0] *= zoomx
            seq_info['intrinsics'][i,1] *= zoomy
            data.cam2_files[i] = np.array(new_filename)
            i+=1
        i = 0
        for filename, output in zip(data.cam3_files, executor.map(load_image, data.cam3_files)):
            img, zoomx, zoomy, orig_img_width, orig_img_height = output
            new_filename = os.path.join(target_dir, filename.split(args.source_dir)[1]).replace('data/','').replace('.png','.jpg').replace('/'+date+'/','/')
            imageio.imwrite(new_filename, img)
            data.cam3_files[i] = np.array(new_filename)
            i+=1

    seq_info['cam_02'] = np.array(data.cam2_files)
    seq_info['cam_03'] = np.array(data.cam3_files)
        
        ###Import libviso2 estimate for correcting
    mono_libviso2_data = sio.loadmat(mono_libviso2_dir+date+'_drive_'+drive+'.mat')  #sparse VO
    stereo_libviso2_data = sio.loadmat(stereo_libviso2_dir+date+'_drive_'+drive+'.mat')  #sparse VO
    mono_traj = mono_libviso2_data['poses_est'].transpose(2,0,1)
    stereo_traj = stereo_libviso2_data['poses_est'].transpose(2,0,1)
        ### store the ground truth pose
    seq_info['sparse_gt_pose'] = stereo_libviso2_data['poses_gt'].transpose(2,0,1)

    mono_seq_info = seq_info.copy()
    stereo_seq_info = seq_info.copy()
    
    ### store the VO pose estimates to extract 
    mono_seq_info['sparse_vo'] = mono_traj
    stereo_seq_info['sparse_vo'] = stereo_traj
    
        ###filter out frames with low rotational or translational velocities
    for seq_info in [mono_seq_info, stereo_seq_info]:
        if args.remove_static:
            print("Removing Static frames from {}".format(drive))            
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

    
    sio.savemat(seq_dir + '/mono_data.mat', mono_seq_info)
    sio.savemat(seq_dir + '/stereo_data.mat', stereo_seq_info)
