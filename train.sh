#!/bin/sh
# about your tracker
script="vit_dist"
config="vit_256_h128_noKD"
num_gpus=1
num_thread=8
# training
python tracking/train.py --script $script --config $config --save_dir ./output --mode multiple --nproc_per_node $num_gpus --use_wandb 0
# evaluation for lasot
python tracking/test.py vit_dist vit_256_h128_noKD --dataset lasot --threads $num_thread --num_gpus $num_gpus
python tracking/analysis_results.py
# evaluation for got10k
python tracking/test.py vit_dist vit_256_h128_noKD --dataset got10k_test --threads $num_thread --num_gpus $num_gpus
python lib/test/utils/transform_got10k.py --tracker_name vit_dist --cfg_name vit_256_h128_noKD
# evaluation for trackingnet
python tracking/test.py vit_dist vit_256_h128_noKD --dataset trackingnet --threads $num_thread --num_gpus $num_gpus
python lib/test/utils/transform_trackingnet.py --tracker_name vit_dist --cfg_name vit_256_h128_noKD
