#!/bin/bash
#
## about your tracker
#cd HiT
#script="HiT"
#config="HiT_Base"
num_gpus=2
num_thread=8
#
## evaluation for got10k
##python tracking/test.py $script $config --dataset got10k_test --threads $num_thread --num_gpus $num_gpus
##python lib/test/utils/transform_got10k.py --tracker_name $script --cfg_name $config
## evaluation for trackingnet
#python tracking/test.py $script $config --dataset trackingnet --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_trackingnet.py --tracker_name $script --cfg_name $config
## evaluation for lasot
#python tracking/test.py $script $config --dataset lasot --threads $num_thread --num_gpus $num_gpus
#python tracking/analysis_results.py
#cd ..


# about your tracker
script="vit_dist"
config="vit_48_h32_noKD"

## evaluation for got10k
#python tracking/test.py $script $config --dataset got10k_test --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_got10k.py --tracker_name $script --cfg_name $config
## evaluation for trackingnet
#python tracking/test.py $script $config --dataset trackingnet --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_trackingnet.py --tracker_name $script --cfg_name $config
# evaluation for lasot
python tracking/test.py $script $config --dataset lasot --threads $num_thread --num_gpus $num_gpus
python tracking/analysis_results.py

# about your tracker
#script="ostrack"
#config="vitb_td_osckpt_nosig"

# evaluation for got10k
#python tracking/test.py $script $config --dataset got10k_test --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_got10k.py --tracker_name $script --cfg_name $config
# evaluation for trackingnet
#python tracking/test.py $script $config --dataset trackingnet --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_trackingnet.py --tracker_name $script --cfg_name $config
# evaluation for lasot
#python tracking/test.py $script $config --dataset lasot --threads $num_thread --num_gpus $num_gpus
#python tracking/analysis_results.py