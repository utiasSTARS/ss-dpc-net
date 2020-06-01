#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)

###Stereo Experiments

python3 run_mono_dpc.py --date $d --train_seq 'all' --val_seq '00' --test_seq '05' --exploss --exp_weight 0.23 --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6 --num_epochs 30 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --save_results

python3 run_mono_dpc.py --date $d --train_seq 'all' --val_seq '05' --test_seq '02' --exploss --exp_weight 0.23 --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6 --num_epochs 30 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --save_results

python3 run_mono_dpc.py --date $d --train_seq 'all' --val_seq '05' --test_seq '00' --exploss --exp_weight 0.23 --augment_motion --dropout_prob 0.5 --lr 5e-3 --wd 4e-6 --num_epochs 30 --lr_decay_epoch 4 --normalize_img --use_flow --estimator_type 'stereo' --save_results


