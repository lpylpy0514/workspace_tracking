#!/bin/bash
# preprocess
source /root/anaconda3/bin/activate seqtrackv2
pip install wandb
pip install thop
apt-get install libturbojpeg
cd /18353470163/lpy/workspace_tracking
# about your tracker
script="ostrack"
config="vitb_ce_te"
#config="spd_test"
num_gpus=2
num_thread=8

# training
python tracking/train.py --script $script --config $config --save_dir ./output --mode multiple --nproc_per_node $num_gpus --use_wandb 0

