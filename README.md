# Workspace for Track

## Install the environment
**Option1**: Use the Anaconda (CUDA 11.1)

```
conda create -n py38 python=3.8
conda activate py38
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
sh install.sh
```


## Set project paths
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${PROJECT_ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- coco
            |-- annotations
            |-- images
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```


## Training
OSTrack: Download pre-trained [MAE ViT-Base weights](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put it under `$PROJECT_ROOT$/pretrained_models` (different pretrained models can also be used, see [MAE](https://github.com/facebookresearch/mae) for more details).

```
python tracking/train.py --script ostrack --config vitb_256_mae_ce_32x4_ep300 --save_dir ./output --mode multiple --nproc_per_node 1 --use_wandb 0
```

Replace `--config` with the desired model config under `experiments/ostrack`. We use [wandb](https://github.com/wandb/client) to record detailed training logs, in case you don't want to use wandb, set `--use_wandb 0`.


## Evaluation
Download the model weights from [Google Drive](https://drive.google.com/drive/folders/1PS4inLS8bWNCecpYZ0W2fE5-A04DvTcd?usp=sharing) 

Put the downloaded weights on `$PROJECT_ROOT$/output/checkpoints/train/ostrack`

Change the corresponding values of `lib/test/evaluation/local.py` to the actual benchmark saving paths

Some testing examples:
- LaSOT or other off-line evaluated benchmarks (modify `--dataset` correspondingly)
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset lasot --threads 16 --num_gpus 4
python tracking/analysis_results.py # need to modify tracker configs and names
```
- GOT10K-test
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_got10k_ep100 --dataset got10k_test --threads 16 --num_gpus 4
python lib/test/utils/transform_got10k.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_got10k_ep100
```
- TrackingNet
```
python tracking/test.py ostrack vitb_384_mae_ce_32x4_ep300 --dataset trackingnet --threads 16 --num_gpus 4
python lib/test/utils/transform_trackingnet.py --tracker_name ostrack --cfg_name vitb_384_mae_ce_32x4_ep300
```

## Visualization or Debug 
[Visdom](https://github.com/fossasia/visdom) is used for visualization. 
1. Alive visdom in the sever by running `visdom`:

2. Simply set `--debug 1` during inference for visualization, e.g.:
```
python tracking/test.py ostrack ostrack384_elimination_ep300 --dataset vot22 --threads 1 --num_gpus 1 --debug 1
```
3. Open `http://localhost:8097` in your browser (remember to change the ip address and port according to the actual situation).

4. Then you can visualize the candidate elimination process.

![ECE_vis](https://github.com/botaoye/OSTrack/blob/main/assets/vis.png)


## Test FLOPs, and Speed
*Note:* The speeds reported in our paper were tested on a single RTX2080Ti GPU.

```
# Profiling vitb_256_mae_ce_32x4_ep300
python tracking/profile_model.py --script ostrack --config vitb_256_mae_ce_32x4_ep300
# Profiling vitb_384_mae_ce_32x4_ep300
python tracking/profile_model.py --script ostrack --config vitb_384_mae_ce_32x4_ep300
```


## Acknowledgments
* Thanks for the [STARK](https://github.com/researchmm/Stark) and [PyTracking](https://github.com/visionml/pytracking) library, which helps us to quickly implement our ideas.
* We use the implementation of the ViT from the [Timm](https://github.com/rwightman/pytorch-image-models) repo.  

patch embedding -> levit
LN -> BN
MTR
KD
FAF
GA
