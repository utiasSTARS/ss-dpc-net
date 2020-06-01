#!/usr/bin/env bash
d=$(date +%Y%m%d%H%M)

###Monocular Experiments 

	#Online
python3 run_mono_dpc.py --date $d --exploss --train_seq 'all' --val_seq '00' --test_seq '09' --augment_motion  --lr 5e-5 --wd 4e-6  --exp_weight 0.23  --num_epochs 30 --lr_decay_epoch 10 --dropout_prob 0.5 --normalize_img --use_flow --estimator_type 'mono' --save_results

python3 run_mono_dpc.py --date $d --exploss --train_seq 'all' --val_seq '05' --test_seq '10' --augment_motion  --lr 5e-5 --wd 4e-6  --exp_weight 0.23  --num_epochs 30 --lr_decay_epoch 10 --dropout_prob 0.5 --normalize_img --use_flow --estimator_type 'mono' --save_results


