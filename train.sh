#!/bin/bash
# preprocess
#source /root/anaconda3/bin/activate seqtrackv2
#pip install wandb
#pip install thop
#cd /18353470163/lpy/workspace_tracking
# about your tracker
script="ostrack"
config="vitb_templateembedding_b64"
num_gpus=2
num_thread=8

# training
python tracking/train.py --script $script --config $config --save_dir ./output --mode multiple --nproc_per_node $num_gpus --use_wandb 0

# evaluation for got10k
#python tracking/test.py $script $config --dataset got10k_test --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_got10k.py --tracker_name $script --cfg_name $config
## evaluation for trackingnet
#python tracking/test.py $script $config --dataset trackingnet --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_trackingnet.py --tracker_name $script --cfg_name $config
# evaluation for lasot
#python tracking/test.py $script $config --dataset lasot --threads $num_thread --num_gpus $num_gpus
#python tracking/analysis_results.py
