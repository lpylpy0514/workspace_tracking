#!/bin/bash

# about your tracker
script="HiT"
config="HiT_Base"
#config="spd_test"
num_gpus=1
num_thread=8

# training
python tracking/train.py --script $script --config $config --save_dir ./output --mode multiple --nproc_per_node $num_gpus --use_wandb 0

