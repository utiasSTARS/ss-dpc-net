Supporting Code for "Self-Supervised Deep Pose Corrections for Robust Visual Odometry"

<img src="https://github.com/utiasSTARS/ss-dpc-net/blob/master/system.png" width="600px"/>

<img src="https://github.com/utiasSTARS/ss-dpc-net/blob/master/network.png" width="600px"/>

## Dependencies:
* numpy
* scipy
* [pytorch](https://pytorch.org/) 
* [liegroups](https://github.com/utiasSTARS/liegroups)
* [pyslam](https://github.com/utiasSTARS/pyslam)
* [tensorboardX](https://github.com/lanpa/tensorboardX)

# Datasets

We trained and tested on the KITTI dataset. Download the raw dataset [here](http://www.cvlibs.net/datasets/kitti/raw_data.php). We provide a dataloader, but we first require that the data be preprocessed. To do so, run `create_kitti_data.py` within `ss-dpc-net/data` (be sure to specify the source and target directory). We preprocessed the data by resizing the images and removing 'static' frames.

# Training

Two bash scripts are provided that will run the training experiments (for monocular pose corrections and stereo pose corrections respectively):

`run_mono_exps.sh`
`run_stereo_exps.sh`

Prior to training, the data directory should be modified accordingly to point to the processed KITTI data. During training, to visualize the training procedure, open a tensorboard from the main directory:

`tensorboard --logdir runs` 

# Inference

run:

`run_inference.py`

This will recompute the pose corrections for a specified KITTI sequence. Currently, it plots the corrected trajectory only.

# Reproduction of Paper Results

Our pretrained models are available online. To download them, run the following bash script from the source directory:

```
bash download_data.sh
```

Within `paper_plots_and_data`, run the various scripts to generate the tables and/or plot the trajectories within our paper. Our pretrained models are also available and can be run on any of the KITTI odometry sequences using `run_inference.py`.

