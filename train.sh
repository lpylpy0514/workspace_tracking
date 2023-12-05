#!/bin/sh
# about your tracker
#script="vittrack"
#config="d12c256conv2_mtr_standard"
#num_gpus=2
#num_thread=8

### training
#python tracking/train.py --script $script --config $config --save_dir ./output --mode multiple --nproc_per_node $num_gpus --use_wandb 0

### evaluation for got10k
#python tracking/test.py $script $config --dataset got10k_test --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_got10k.py --tracker_name $script --cfg_name $config
### evaluation for trackingnet
#python tracking/test.py $script $config --dataset trackingnet --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_trackingnet.py --tracker_name $script --cfg_name $config
### evaluation for lasot
#python tracking/test.py $script $config --dataset lasot --threads $num_thread --num_gpus $num_gpus
#python tracking/analysis_results.py


script="ostrack"
config="vitb_templateembedding"
#config="vitb_256_mae_32x4_ep300"
num_gpus=2
num_thread=8

# training
python tracking/train.py --script $script --config $config --save_dir ./output --mode multiple --nproc_per_node $num_gpus --use_wandb 0

## evaluation for got10k
#python tracking/test.py $script $config --dataset got10k_test --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_got10k.py --tracker_name $script --cfg_name $config
## evaluation for trackingnet
#python tracking/test.py $script $config --dataset trackingnet --threads $num_thread --num_gpus $num_gpus
#python lib/test/utils/transform_trackingnet.py --tracker_name $script --cfg_name $config
## evaluation for lasot
#python tracking/test.py $script $config --dataset lasot --threads $num_thread --num_gpus $num_gpus
#python tracking/analysis_results.py

#scp -r b507@10.7.63.141:/home/b507/Downloads/data/TNL2K_TEST /home/ymz/newdisk1
