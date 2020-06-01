import cv2
import pykitti
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import scipy.io as sio
from liegroups import SE3, SO3
import os
import glob

class KittiLoaderPytorch(torch.utils.data.Dataset):
    """Loads the KITTI Odometry Benchmark Dataset"""
    def __init__(self, basedir, config, seq, mode='train', transform_img=None, num_frames=None, augment=False, skip=None, augment_backwards=False):
        """
        Args:
            directory to processed KITTI sequences
            config file
            desired sequences (use 'all' to load all sequences into the training dataset, except for the specified val and test sets)
            transform to apply to the images in the dataset
        """
        seq_names= {'00': '2011_10_03_drive_0027_sync',
                '01': '2011_10_03_drive_0042_sync',
                '02': '2011_10_03_drive_0034_sync',
                '04': '2011_09_30_drive_0016_sync',
                '05': '2011_09_30_drive_0018_sync',
                '06': '2011_09_30_drive_0020_sync',
                '07': '2011_09_30_drive_0027_sync',
                '08': '2011_09_30_drive_0028_sync',
                '09': '2011_09_30_drive_0033_sync',
                '10': '2011_09_30_drive_0034_sync',
                '11': '11',
                '12': '12',
                '13': '13',
                '14': '14',
                '15': '15',
                '16': '16',
                '17': '17',
                '18': '18',
                '19': '19',
                '20': '20',
                '21': '21',
            
                }

        self.config = config
        self.seq_len = config['img_per_sample']
        self.transform_img = transform_img
        self.num_frames = num_frames
        self.augment = augment
        self.skip = skip
        self.augment_backwards = augment_backwards

            ###Iterate through all specified KITTI sequences and extract raw data, and trajectories
        self.left_cam_filenames = []
        self.right_cam_filenames = []
        self.raw_intrinsic_trials = []
        self.raw_gt_trials = []
        self.raw_vo_traj = []
        train_seq, val_seq, test_seq = seq
        if train_seq == ['all'] and mode == 'train':
            seq = []
            seq_name={}

            for d in glob.glob('{}/**'.format(basedir), recursive=False):
                name = d.replace(basedir, '').replace('/','')
                i=0
                for s in val_seq:
                    if name == seq_names[s]:
                        i=1
                        print("excluding {} from training data (it's a validation sequence)".format(name))
                for s in test_seq:
                    if name == seq_names[s]:
                        i=1
                        print("excluding {} from training data (it's a test sequence)".format(name))                    
                if i == 0:
                    seq.append(name)
                    seq_name[name] = name
        if train_seq != ['all'] and mode == 'train':
            seq = train_seq
            seq_name = seq_names
        if mode == 'val':
            seq_name = seq_names
            seq = val_seq
        if mode == 'test':
            seq_name = seq_names
            seq = test_seq
        print('{} sequences: {}'.format(mode,seq))
        for s,i in zip(seq,range(0,len(seq))):
            data = sio.loadmat(os.path.join(basedir, seq_name[s],'{}_data.mat'.format(config['estimator_type'])))
            
            self.left_cam_filenames.append(np.copy(data['cam_02'].reshape((-1,1))))
            self.right_cam_filenames.append(np.copy(data['cam_03'].reshape((-1,1))))
            self.raw_intrinsic_trials.append(np.copy(data['intrinsics']))
            self.raw_gt_trials.append(np.copy(data['sparse_gt_pose']))
            self.raw_vo_traj.append(np.copy(data['sparse_vo']))
            
            if self.config['correction_rate'] != 1:
                cr = self.config['correction_rate']
                self.left_cam_filenames[i] = np.copy(self.left_cam_filenames[i][::cr])
                self.right_cam_filenames[i] = np.copy(self.right_cam_filenames[i][::cr])
                self.raw_intrinsic_trials[i] = np.copy(self.raw_intrinsic_trials[i][::cr])
                self.raw_gt_trials[i] = np.copy(self.raw_gt_trials[i][::cr])
                self.raw_vo_traj[i] = np.copy(self.raw_vo_traj[i][::cr])
            else:
                cr=1
            
            if self.num_frames:
                self.left_cam_filenames[i] = np.copy(self.left_cam_filenames[i][:self.num_frames])
                self.right_cam_filenames[i] = np.copy(self.right_cam_filenames[i][:self.num_frames])
                self.raw_intrinsic_trials[i] = np.copy(self.raw_intrinsic_trials[i][:self.num_frames])
                self.raw_gt_trials[i] = np.copy(self.raw_gt_trials[i][:self.num_frames])
                self.raw_vo_traj[i] = np.copy(self.raw_vo_traj[i][:self.num_frames])       

            
            #also add a trial that skips every other frame, for augmented data that simulates faster motion.
        if self.augment == True:
            for s, i in zip(seq, range(0,len(seq))):
                data = sio.loadmat(os.path.join(basedir, seq_name[s],'{}_data.mat'.format(config['estimator_type'])))
                self.left_cam_filenames.append(np.copy(data['cam_02'].reshape((-1,1)))[::(cr+1)])
                self.right_cam_filenames.append(np.copy(data['cam_03'].reshape((-1,1)))[::(cr+1)])
                self.raw_intrinsic_trials.append(np.copy(data['intrinsics'])[::(cr+1)])
                self.raw_gt_trials.append(np.copy(data['sparse_gt_pose'])[::(cr+1)])
                self.raw_vo_traj.append(np.copy(data['sparse_vo'])[::(cr+1)])
         
        if self.augment_backwards:
            for s, i in zip(seq, range(0,len(seq))):
                self.left_cam_filenames.append(np.flip(np.copy(self.left_cam_filenames[i]),axis=0))
                self.right_cam_filenames.append(np.flip(np.copy(self.right_cam_filenames[i]),axis=0))
                self.raw_intrinsic_trials.append(np.flip(np.copy(self.raw_intrinsic_trials[i]),axis=0))
                self.raw_gt_trials.append(np.flip(np.copy(self.raw_gt_trials[i]),axis=0))
                self.raw_vo_traj.append(np.flip(np.copy(self.raw_vo_traj[i]),axis=0))   
                
###         Merge data from all trials   
        self.gt_samples, self.left_img_samples, self.right_img_samples, self.intrinsic_samples, self.vo_samples, = self.reshape_data()

        if self.skip:
            skip = self.config['skip']
            self.gt_samples, self.left_img_samples, self.intrinsic_samples = self.gt_samples[::skip], self.left_img_samples[::skip], self.intrinsic_samples[::skip]
            self.vo_samples = self.vo_samples[::skip]
            self.right_img_samples = self.right_img_samples[::skip]


    def __len__(self):
        return int(self.gt_samples.shape[0])
  
    def __getitem__(self, idx):
        imgs = []
        for i in range(0,self.seq_len):
            imgs.append(self.load_image(self.left_img_samples[idx,i]))
         
        imgs = list(imgs)
        intrinsics = self.intrinsic_samples[idx] 
        gt_lie_alg, vo_lie_alg, gt_correction = self.compute_target(idx)
        orig_imgs = imgs.copy()
        if self.transform_img != None:
            imgs, intrinsics, gt_lie_alg = self.transform_img(imgs, intrinsics, gt_lie_alg)          
        if self.config['use_flow']:
            for i in range(0,len(imgs)-1):  
                flow_img1 = np.array(Image.fromarray(orig_imgs[i]).convert('L'))
                flow_img2 = np.array(Image.fromarray(orig_imgs[i+1]).convert('L'))
                flow_img = cv2.calcOpticalFlowFarneback(flow_img1,flow_img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_img = torch.from_numpy(np.transpose(flow_img, (2,0,1))).float()
                orig_imgs = [torch.from_numpy(np.transpose(im, (2, 0, 1))).float()/255 for im in orig_imgs]
                imgs.append(flow_img)
                imgs.append(orig_imgs[0]) #stack regular imgs at end for loss evaluation (keeps things consistent)
                imgs.append(orig_imgs[1])

        return imgs, gt_lie_alg, intrinsics, vo_lie_alg, gt_correction

    def load_image(self, img_file):
        img = np.array(Image.open(img_file[0]))
        return img
    
    def compute_target(self, idx):
        #compute Tau_gt
        T2 = SE3.from_matrix(self.gt_samples[idx,1,:,:],normalize=True).inv()
        T1 = SE3.from_matrix(self.gt_samples[idx,0,:,:],normalize=True)
        dT_gt = T2.dot(T1)
        gt_lie_alg = dT_gt.log()
         
        #compute Tau_vo
        T2 = SE3.from_matrix(self.vo_samples[idx,1,:,:],normalize=True).inv()
        T1 = SE3.from_matrix(self.vo_samples[idx,0,:,:],normalize=True)
        dT_vo = T2.dot(T1)
        vo_lie_alg = dT_vo.log()
        
        if self.config['estimator_type'] == 'mono':
            if np.linalg.norm(vo_lie_alg[0:3]) >= 1e-8:
                scale = np.linalg.norm(gt_lie_alg[0:3])/np.linalg.norm(vo_lie_alg[0:3])
                vo_lie_alg[0:3] = scale*vo_lie_alg[0:3]

        gt_correction = (dT_gt.dot(dT_vo.inv())).log()
        return gt_lie_alg, vo_lie_alg, gt_correction   

    def reshape_data(self):
        self.samples_per_file=None
        gt_samples = self.raw_gt_trials[0][0:self.seq_len,:,:].reshape((1,self.seq_len,4,4))
        vo_samples = self.raw_vo_traj[0][0:self.seq_len,:,:].reshape((1,self.seq_len,4,4))
        intrinsic_samples = self.raw_intrinsic_trials[0][0:self.seq_len,:,:].reshape((1,self.seq_len,3,3))
        left_img_samples = self.left_cam_filenames[0][0:self.seq_len].reshape((1,self.seq_len,1))
        right_img_samples = self.right_cam_filenames[0][0:self.seq_len].reshape((1,self.seq_len,1))
        pose_per_sample = []

        for gt in self.raw_gt_trials:
            gt = gt.reshape((-1,gt.shape[1]*gt.shape[2]))
            if self.samples_per_file == None:
                num_samples = int(gt.shape[0]/(self.seq_len))
            new_gt = self.split_data(gt, num_samples, self.seq_len)
            new_gt = new_gt.reshape((-1,self.seq_len,4,4))
            gt_samples = np.vstack((gt_samples, new_gt))
            pose_per_sample.append(new_gt.shape[0])
            
        for vo in self.raw_vo_traj:
            vo = vo.reshape((-1,vo.shape[1]*vo.shape[2]))
            if self.samples_per_file == None:
                num_samples = int(vo.shape[0]/(self.seq_len))
            new_vo = self.split_data(vo, num_samples, self.seq_len)
            new_vo = new_vo.reshape((-1,self.seq_len,4,4))
            vo_samples = np.vstack((vo_samples, new_vo))
            
        for intrins in self.raw_intrinsic_trials:
            intrins = intrins.reshape((-1,intrins.shape[1]*intrins.shape[2]))
            if self.samples_per_file == None:
                num_samples = int(intrins.shape[0]/(self.seq_len))
            new_intrins = self.split_data(intrins, num_samples, self.seq_len)
            new_intrins = new_intrins.reshape((-1,self.seq_len,3,3))
            intrinsic_samples = np.vstack((intrinsic_samples, new_intrins))
        
        count = 0
        for im in self.left_cam_filenames:
            im = im.reshape((-1,1))
            if self.samples_per_file == None:
                num_samples = int(im.shape[0]/(self.seq_len))
            new_imgs = self.split_data(im, num_samples, self.config['img_per_sample'])
            left_img_samples = np.vstack((left_img_samples, new_imgs[0:pose_per_sample[count]])) #get rid of extra imgs due to rounding of gt
            count +=1

        count = 0
        for im in self.right_cam_filenames:
            im = im.reshape((-1,1))
            if self.samples_per_file == None:
                num_samples = int(im.shape[0]/(self.seq_len))
            new_imgs = self.split_data(im, num_samples, self.config['img_per_sample'])
            right_img_samples = np.vstack((right_img_samples, new_imgs[0:pose_per_sample[count]])) #get rid of extra imgs due to rounding of gt
            count +=1
           
        gt_samples = gt_samples[1:]
        intrinsic_samples = intrinsic_samples[1:]
        left_img_samples = left_img_samples[1:]
        right_img_samples = right_img_samples[1:]
        vo_samples = vo_samples[1:]
     
        return gt_samples, left_img_samples, right_img_samples, intrinsic_samples, vo_samples
        
    def split_data(self, data, num_samples,sample_size):
        Type = data.dtype
        samplesize=int(sample_size)
        output = np.zeros((1,samplesize,data.shape[1])).astype(Type)
        i=0
        while i <= (data.shape[0]-sample_size):
            output = np.vstack((output, data[i:i+samplesize].reshape(1,samplesize,data.shape[1])))
            i+=(samplesize-1)            
        return output[1:] 
