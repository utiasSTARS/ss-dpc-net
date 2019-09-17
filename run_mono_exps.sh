#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)

###Monocular Experiments 

	#Online
python3 run_mono_dpc.py --data_dir '/media/m2-drive/datasets/KITTI-downsized' --date $d --exploss --train_seq 'all' --val_seq '00' --test_seq '09' --augment_motion  --lr 5e-5 --wd 4e-6  --exp_weight 0.23  --num_epochs 25 --lr_decay_epoch 10 --dropout_prob 0.5 --normalize_img --use_flow --estimator_type 'mono' --mode 'online' --save_results


	#Offline

#python3 run_mono_dpc.py --date $d --data_dir '/media/m2-drive/datasets/KITTI-downsized' --exploss --exp_weight 0.23 --train_seq '09' --val_seq '09' --test_seq '09' --dropout_prob 0.0 --lr 1e-3 --wd 0 --lr_decay_epoch 15 --num_epochs 75 --estimator_type 'mono' --mode 'offline' --save_results



