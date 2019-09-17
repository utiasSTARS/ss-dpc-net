#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)

###Stereo Experiments

	#Online
#python3 run_mono_dpc.py --date $d --train_seq 'all' --val_seq '00' --test_seq '05' --exploss --exp_weight 0.23 --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6 --num_epochs 25 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --mode 'online' --save_results
#python3 run_mono_dpc.py --date $d --train_seq 'all' --val_seq '05' --test_seq '02' --exploss --exp_weight 0.23 --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6 --num_epochs 25 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --mode 'online' --save_results
#python3 run_mono_dpc.py --date $d --train_seq 'all' --val_seq '05' --test_seq '00' --exploss --exp_weight 0.23 --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6 --num_epochs 25 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --mode 'online' --save_results

#python3 run_mono_dpc.py --data_dir '/media/m2-drive/datasets/KITTI-downsized' --date $d --exploss --train_seq 'all' --val_seq '00' --test_seq '02' --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6  --exp_weight 0.23 --num_epochs 25 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --mode 'online' --save_results

#python3 run_mono_dpc.py --date $d --exploss --train_seq 'all' --val_seq '05' --test_seq '06' --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6  --exp_weight 0.23 --num_epochs 25 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --mode 'online' --save_results

#python3 run_mono_dpc.py --date $d --exploss --train_seq 'all' --val_seq '00' --test_seq '07' --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6  --exp_weight 0.23 --num_epochs 25 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --mode 'online' --save_results

#python3 run_mono_dpc.py --date $d --exploss --train_seq 'all' --val_seq '00' --test_seq '08' --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6  --exp_weight 0.23 --num_epochs 25 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --mode 'online' --save_results

#python3 run_mono_dpc.py --date $d --exploss --train_seq 'all' --val_seq '00' --test_seq '09' --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6  --exp_weight 0.23 --num_epochs 25 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --mode 'online' --save_results

#python3 run_mono_dpc.py --date $d --exploss --train_seq 'all' --val_seq '00' --test_seq '10' --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6  --exp_weight 0.23 --num_epochs 25 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --mode 'online' --save_results



	#Offline
python3 run_mono_dpc.py --date $d --data_dir '/media/m2-drive/datasets/KITTI-downsized' --mode 'offline' --estimator_type 'stereo' --exploss --exp_weight 0.23 --train_seq '02' --val_seq '02' --test_seq '02' --dropout_prob 0.0 --lr 1e-3 --wd 0 --lr_decay_epoch 15 --num_epochs 75 --save_results

