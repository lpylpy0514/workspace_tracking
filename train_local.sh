#!/bin/bash

# about your tracker
script="ostrack"
config="vitb_debug"
#config="spd_test"
num_gpus=2
num_thread=8

# training
python tracking/train.py --script $script --config $config --save_dir ./output --mode multiple --nproc_per_node $num_gpus --use_wandb 0

