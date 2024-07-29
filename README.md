# IMUDB

Here's the official implementation of the accepted RA-L paper: 
[K. Yuan and Z. J. Wang, "A Simple Self-Supervised IMU Denoising Method For Inertial Aided Navigation," in IEEE Robotics and Automation Letters, doi: 10.1109/LRA.2023.3234778.](https://ieeexplore.ieee.org/document/10008040)

![imudb-demo](https://github.com/KleinYuan/IMUDB/assets/8921629/772efc08-855f-48a6-a043-aa175d553327)


## Project Walk-Thru

This project: 

- is based on [pytorch-lightning](https://www.pytorchlightning.ai/), and each module of this project
is compatible with and can be directly reused by native pytorch projects.
- is dockerized so that you should be able to run it on many platforms
- refers to my previous works [RAL@2020](https://github.com/KleinYuan/RGGNet) and [TRO@2022](https://github.com/KleinYuan/LiCaS3) in terms of environment setups.
The main difference is that previous works use tensorflow 1.x and this one uses pytorch-lightning.


The core structure is as follows:

- [configs](configs): yaml configuration files
- [data](data): a wrapper to create training/validation/testing data splits with dataloaders. This can pbbly be merged into the dataloader 
in later refactoring
- [dataloaders](dataloaders): parsers, classes of constructing torch.utils.data.dataset.Dataset from raw data
- [models](models): actual neural networks forwards/losses/optimizer/tensorboard_metrics
- [networks](networks): more like a library of building blocks. We may name this better in later refactoring.
- [experiments](experiments): actual experiments of training and evaluations


## Hardware Requirements

I conducted my experiments with the following hardware setups, which is highly recommended but not necessarily a hard
requirement.

| Item | Version|
|---|---|
| System | 18.04 (x86), referring to [MACHINE-SETUP.md](https://github.com/KleinYuan/RGGNet/blob/master/MACHINE-SETUP.md) to install the system|
| SSD | 2T |
| GPU Memory | >= 11 GB |

## Main Software Requirements

| Item | Version        |
|---|----------------|
| Python | 3.6.x          |
| NVIDIA-Driver | 440.36         |
| CUDA  | 10.2           |
| cuDNN| v7.4.2         |
| Pytorch | 1.8 |
| Pytorch-Lightning | 1.4.2           |

All training and most evaluation will happen inside docker so that the environment is transferable.

You can refer to [Dockerfile](Dockerfile) and [requirements.txt](requirements.txt) for more details.

## Prepare Raw Datasets

In our work, [EuRoC and TUM-VI](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) are used to benchmark the performances.
Both datasets can be downloaded from the official website: https://vision.in.tum.de/data/datasets/visual-inertial-dataset

Below is the folder structure we use for EuRoC:

```
/${PWD}/EuRoC/
├── bags
│   ├── MH_02_easy.bag
│   ├── MH_04_difficult.bag
│   ├── V1_01_easy.bag
│   ├── V1_03_difficult.bag
│   ├── V2_02_medium.bag
├── Preload
├── Raw
│   ├── MH_01_easy
│   ├── MH_02_easy
│   ├── MH_03_medium
│   ├── MH_04_difficult
│   ├── MH_05_difficult
│   ├── V1_01_easy
│   ├── V1_02_medium
│   ├── V1_03_difficult
│   ├── V2_01_easy
│   ├── V2_02_medium
│   └── V2_03_difficult
```

For training, we will only need to use the Raw data in the "Raw" folder. The "bags" folder contains all the raw [ROS](https://www.ros.org/) bag files,
which will only be used for testing/evaluation.
Please also create an empty folder "Preload" manually to contain the caches of training data. This will largely accelerate the training
process.

Similarly, below is the folder structure for TUM-VI dataset:

```
/${PWD}/TUMVI/
├── bags
│   ├── dataset-outdoors2_512_16.bag
│   ├── dataset-outdoors4_512_16.bag
│   ├── dataset-outdoors6_512_16.bag
├── Preload
└── Raw
    ├── dataset-outdoors1_512_16
    ├── dataset-outdoors2_512_16
    ├── dataset-outdoors3_512_16
    ├── dataset-outdoors4_512_16
    ├── dataset-outdoors5_512_16
    ├── dataset-outdoors6_512_16
    ├── dataset-outdoors7_512_16
    ├── dataset-outdoors8_512_16
    ├── dataset-room1_512_16
    ├── dataset-room2_512_16
    ├── dataset-room3_512_16
    ├── dataset-room4_512_16
    ├── dataset-room5_512_16
    └── dataset-room6_512_16
```

which is the same story.


## Training

Note: all below happens inside the docker container, so that you can pbbly run it everywhere.

### Build the Image and Enter a Shell

Build the docker image: 

```
make build
```

Update the volume mount of your training data in the [Makefile](Makefile):

```
run:
	sudo docker run -it \
        --runtime=nvidia \
        --name="imudb-experiment-gpu" \
        --net=host \
        --privileged=true \
        --ipc=host \
        --memory="20g" \
        --memory-swap="20g" \
        -v ${PWD}:/root/imudb \
        -v  ${PWD}/EuRoC:/root/EuRoC \
        -v  ${PWD}/TUMVI:/root/TUMVI \
      	imudb-docker bash
```

I by default believe that you put EuRoC and TUMVI data under `${PWD}`. However, if that's not true, update it.

At last, enter a shell with simply doing the following:

```
make run
```

And then you shall enter a shell with exact same environment of what I was using!

Be aware that I use a fixed name for the container and if you exit the docker with it pending, you may not be able to 
run `make run` again. In case you are not familiar with docker, you can simply do `make clean`, which will remove all the 
containers.



### Run Training

For who just want to reproduce our experiments without digging into deep, we prepare a [Makefile](Makefile) for you, with many
one line command to run it:

```
train-euroc-limu-bert:
	export PYTHONPATH='.' && python experiments/train_euroc_limu_bert.py

train-euroc-imudb:
	export PYTHONPATH='.' && python experiments/train_euroc_imudb.py

train-tumvi-imudb:
	export PYTHONPATH='.' && python experiments/train_tumvi_imudb.py
```

For example, if you would like to run training for IMUDB against EuRoC, then simply type:

```
make train-euroc-imudb
```

And you shall see something like this in the terminal log:

```
export PYTHONPATH='.' && python experiments/train_euroc_imudb.py
Global seed set to 1234
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

  | Name          | Type                   | Params
---------------------------------------------------------
0 | limu_bert_mlm | LIMUBertModel4Pretrain | 62.6 K
1 | limu_bert_nsp | LIMUBertModel4Pretrain | 62.6 K
---------------------------------------------------------
125 K     Trainable params
0         Non-trainable params
125 K     Total params
0.501     Total estimated model params size (MB)
Validation sanity check: 0it [00:00, ?it/s]/usr/local/lib/python3.6/dist-packages/pytorch_lightning/trainer/data_loading.py:373: UserWarning: Your val_dataloader has `shuffle=True`, it is best practice to turn this off for val/test/predict dataloaders.
  f"Your {mode}_dataloader has `shuffle=True`, it is best practice to turn"
Global seed set to 1234                                                                          
Epoch 0:  11%|##8                        | 28/262 [00:04<00:38,  6.05it/s, loss=0.39, v_num=8:00]
```

And you will see a folder created called `logs_${your_machine_host_name}` with the following example structure:

```
└── euroc_imudb
    └── dev_imudb_2022-12-27T15:16:19.040391-08:00
        ├── events.out.tfevents.1672182979.2080ti.14.0
        └── hparams.yaml
```

As you can find in the script [experiments/train_euroc_imudb.py](experiments/train_euroc_imudb.py), we create a dedicated 
folder encoded with the experiment timestamp which contains the log and hyper-parameters files. By doing this,
you can easily manage many experiments regardless of the models.

In addition, you will also find a new record in [experiments/experiment_management_log.csv](experiments/experiment_management_log.csv),
which record the information of this experiment. It may make your life easier in your early development stage.

After one epoch, you will find another folder called `checkpoints_${your_machine_host_name}`, containing your trained checkpoints.


### Different Models

In our paper, for EuRoC, we have trained three models: IMUDB, IMUDB without FIF and LIMU-BERT.

The following is a brief guide to train the corresponding models for EuRoC:

- euroc-imudb: `python experiments/train_euroc_imudb.py`
- euroc-imudb without FIF: first update the `nsp_loss_weights` field in the 
[configs/euroc_imudb.yaml](configs/euroc_imudb.yaml) to 0 and then do `python experiments/train_euroc_imudb.py`
- euroc-limu-bert: `python experiments/train_euroc_limu_bert.py`


## Evaluations


First of all, you will need to make sure that you have downloaded the ROS Bags of 
EuRoC/TUM-VI from https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets. 

Second, install [Open-VINS](https://github.com/rpng/open_vins/) and try to run the baseline:

- Step1: install the open vins on your machine, following [this tutorial](https://docs.openvins.com/gs-installing.html)
- Step2: edit the [pgeneva_serial_eth.launch](https://github.com/rpng/open_vins/tree/develop_v2.4/ov_msckf/launch) located in the open_vins folder
to update `bag` field with your actual bag path. This is being said that you may need to download the bags, via `bash tools/download_euro_test_bags.sh` 
(you may need to update where you store the bags).
- Step3: `source devel/setup.bash`
- Step4: `roslaunch ov_msckf pgeneva_serial_eth.launch` will let launch the benchmark and you can visualize it with the [rviz](euroc.rviz)

I added a copy of the [pgeneva_serial_eth.launch](./ros_packages_configs/pgeneva_serial_eth.launch) I used in case you missed it.

Please be noted that the above happens at the host machine instead of the docker container. And below should happen inside
the container.

Third, try to evaluate IMUDB empowered one:

- Generate IMUDB empowered bag file (taking `V1_01_easy.bag` as an example), with command such as:
```
python experiments/ros_euroc_imudb_pp.py  process_a_bag_with_ckpts \
--bag_fp=/root/EuRoC/bags/V1_01_easy.bag \
--config_fp=logs_1080tis/euroc_imudb/dev_imudb_2022-02-20T00:29:37.077961-08:00/hparams.yaml \
--ckpts_fp=checkpoints_1080tis/euroc_imudb/dev_imudb_2022-02-20T00:29:37.077961-08:00/euroc_imudb-epoch=3184-val_denoise_loss=0.000005.ckpt
```

- Do the step2 and step4 above again with the new bag. You can use [euroc.rivz](euroc.rviz) or [euroc_paper.rivz](euroc_paper.rviz) for visualization. The later
one is exactly the rivz I used to build the visualization in my paper.

For limu_bert you can also technically just use the same but the correct checkpoint, with one thing being noticed: the uncertainty calculation.
But to make it not complicated, I would just use the following example command for IMUDB without FIF:

```
python experiments/ros_euroc_limu_bert.py  process_a_bag_with_ckpts \
--bag_fp=/root/EuRoC/bags/MH_02_easy.bag \
--config_fp=logs_2080ti/euroc_limu_bert/dev_limu_bert_2022-02-07T23:50:30.791880-08:00/hparams.yaml \
--ckpts_fp=checkpoints_2080ti/euroc_limu_bert/dev_limu_bert_2022-02-07T23:50:30.791880-08:00/euroc_limu_bert-epoch=4509-val_denoise_loss=0.000001.ckpt
```

And the following for LIMU-BERT:

```
python experiments/ros_euroc_limu_bert.py  process_a_bag_with_ckpts \
--bag_fp=/root/EuRoC/bags/V2_02_medium.bag \
--config_fp=logs_2080ti/euroc_limu_bert/dev_limu_bert_2022-02-10T23:07:59.874966-08:00/hparams.yaml \
--ckpts_fp=checkpoints_2080ti/euroc_limu_bert/dev_limu_bert_2022-02-10T23:07:59.874966-08:00/euroc_limu_bert-epoch=3882-val_loss=0.001699.ckpt
```

You can do the same for both EuRoC and TUM-VI.

## Profile the Latency

Below is an example command, which in practice you will need to use your own:

```
python experiments/profile_latency.py  profile_with_a_bag_and_ckpts \
--bag_fp=/root/EuRoC/bags/MH_04_difficult.bag \
--config_fp=logs_1080tis/euroc_imudb/dev_imudb_2022-02-20T00:29:37.077961-08:00/hparams.yaml \
--ckpts_fp=checkpoints_1080tis/euroc_imudb/dev_imudb_2022-02-20T00:29:37.077961-08:00/euroc_imudb-epoch=3100-val_denoise_loss=0.000006.ckpt
```


## Visualization

In our work and many previous works, [evo_traj](https://github.com/MichaelGrupp/evo/wiki/evo_traj) has been
adopted to draw nice trajectories of different methods. This is very straightforward for TUMV-VI dataset results.

For EuRoC-Open-VINS, it's a little bit annoying that the Open-VINS project has its own util tools, which is not as popular as EVO. 
In order to plot nice EVO figures, I made one nice script: [euroc_path_bags_into_csv.py](tools/euroc_path_bags_into_csv.py) 
for it, including the following steps:

- manually record the trajectories of each run: `rosbag record -O open-vins.bag /ov_msckf/pathimu` or `rosbag record -O open-vins.bag /ov_msckf/pathgt`
- convert the imudb, gt and baseline paths into evo-compatible csv: 
```
root@2080ti:~/imudb# python tools/euroc_path_bags_into_csv.py extract_x_y_into_csv  --gt_bag_path ~/EuRoC/Results/MH_04_difficult/gt.bag  --baseline_bag_path ~/EuRoC/Results/MH_04_difficult/open-vins.bag   --imudb_bag_path ~/EuRoC/Results/MH_04_difficult/imudb.bag 
```

And then simply run evo_traj tum mode to draw it!

## Credits

- many thanks to the authors of [LIMU-BERT](https://github.com/dapowan/LIMU-BERT-Public) and many codes are borrowed from that project. Be aware that the code
in this repo has been cleaned when port the native pytorch to pytorch-lightning modules
- many thanks to the authors of [denoise-imu-gyro](https://github.com/mbrossar/denoise-imu-gyro) and many codes are borrowed from that project. Be aware that the code
in this repo has been cleaned when port the native pytorch to pytorch-lightning modules
- many thanks to the authors of [Open-VINS](https://github.com/rpng/open_vins/), whom made the benchmark possible
- many thanks to the authors of [evo_traj](https://github.com/MichaelGrupp/evo/wiki/evo_traj), whom made our visualization life easier
- many thanks to the authors of [pytorch-lightning](https://www.pytorchlightning.ai/), whom makes the code more concise and our research life easier



## Citation

```
@ARTICLE{10008040,
  author={Yuan, Kaiwen and Wang, Z. Jane},
  journal={IEEE Robotics and Automation Letters}, 
  title={A Simple Self-Supervised IMU Denoising Method for Inertial Aided Navigation}, 
  year={2023},
  volume={8},
  number={2},
  pages={944-950},
  doi={10.1109/LRA.2023.3234778}}
```

## Clarification

This repo is a largely refactored open-source version based on my internal experimental repository (which is really messy) for my publications.
If you see potential issues/bugs or have questions regarding my works, please feel free to email me (kaiwen dot yuan1992 at gmail dot com). As I graduated, UBC widthdrew my school email kaiwen@ece.ubc.ca, which is not valid any more.

If you are interested in collaborations with me on related topics, don't hesitate to reach out to me :)
