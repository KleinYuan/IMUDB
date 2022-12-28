# This script deserialize data from a ROS bag, running inference and save the data in a separate rosbags

import yaml
import fire
import rosbag
from collections import deque
import numpy as np
from copy import deepcopy
import multiprocessing
from models.imudb import Model
import torch

"""
# the model_fp can be either onnx or chekpoints
python experiments/ros_euroc.py  process_bags \
--bag_root=/root/EuRoC/bags \
--bag_names="['MH_02_easy.bag', 'MH_04_difficult.bag', 'V2_02_medium.bag', 'V1_03_difficult.bag', 'V1_01_easy.bag']" \
--config_fp=logs_1080ti/euroc/dev_add_noise_and_noise_scale_0.001_self_penalty_2021-10-30T12:31:50.078953-07:00/hparams.yaml \
--model_fp=checkpoints_1080ti/euroc/dev_add_noise_and_noise_scale_0.001_self_penalty_2021-10-30T12:31:50.078953-07:00/euroc-epoch=312-val_loss=0.00.onnx
"""


def process_bags(bag_root, bag_names, model_fp, config_fp):
    if 'onnx' in model_fp:
        raise Exception("Not supported yet")
    else:
        backend = process_a_bag_with_ckpts
    model = backend(None, model_fp, config_fp, None, True)
    jobs = []
    for bag_name in bag_names:
        bag_fp = '{}/{}'.format(bag_root, bag_name)
        # process_a_bag(bag_fp, onnx_fp, config_fp)
        p = multiprocessing.Process(target=backend, args=(bag_fp, model_fp, config_fp, model))
        jobs.append(p)
        p.start()


"""
# Below is the benchmark run in the slides:
python experiments/ros_euroc_imudb.py  process_a_bag_with_ckpts \
--bag_fp=/root/EuRoC/bags/MH_04_difficult.bag \
--config_fp=logs_1080tis/euroc_imudb/dev_imudb_2022-02-20T00:29:37.077961-08:00/hparams.yaml \
--ckpts_fp=checkpoints_1080tis/euroc_imudb/dev_imudb_2022-02-20T00:29:37.077961-08:00/euroc_imudb-epoch=3100-val_denoise_loss=0.000006.ckpt
"""


def process_a_bag_with_ckpts(bag_fp, ckpts_fp, config_fp, external_backend=None, get_model=False):
    REJECT_MSE_THRE = 0.001
    ROUND_DIGIT = 6
    reject_cnt = 0
    msg_cnt = 0

    print("Processing {}....".format(bag_fp))
    with open(config_fp) as f:
        config = yaml.safe_load(f)
    config = config['config']['model']
    print("Loading config ...")
    print(config)

    if external_backend:
        print("Using external backend...")
        backend = external_backend
    else:
        print("Initializing the ckpts_fp session ...")
        model = Model.load_from_checkpoint(ckpts_fp)
        backend = model.limu_bert_mlm.forward
        if get_model:
            return backend

    x_imu_buffer = deque(maxlen=int(config['inputs']['imu_sequence_number']))

    print("Reading bag {}...".format(bag_fp))
    bag = rosbag.Bag(bag_fp)
    new_bag = rosbag.Bag(f"{bag_fp}.new.imudb2.bag", 'w')
    print("Running benchmark ....")
    max_loss = 0
    for topic, msg, t in bag.read_messages():
        if topic == '/cam0/image_raw':
            # https://github.com/eric-wieser/ros_numpy
            # the author of ros_numpy is a hero. Otherwise, I need to sort out the dying python-2 ros-melodic cv_bridge
            # compatible hell with python 3.6.9 pytorch-lightening cuda 10 environment..... :((((((
            new_bag.write(topic, msg, msg.header.stamp)

        elif topic == '/imu0':
            msg_cnt += 1
            # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Imu.html
            x_imu_buffer.append(
                [
                    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
                ]
            ) # (S, 6)
            #print(f"camera buffer length is {len(x_cam_buffer)}, imu buffer length is {len(x_imu_buffer)}")
            new_imu_msg = deepcopy(msg)
            buffer_is_valid = (len(x_imu_buffer) == int(config['inputs']['imu_sequence_number']))

            if buffer_is_valid:
                x_imu_infer = np.array([x_imu_buffer]).astype(np.float32)
                x_imu_infer[:, :, 3:] = x_imu_infer[:, :, 3:] / 9.8
                hat_imu = backend(torch.tensor(x_imu_infer))

                hat_imu = hat_imu.detach().numpy()
                # denorm
                hat_imu = hat_imu[0]
                hat_imu[:, 3:] = hat_imu[:, 3:] * 9.8
                hat_imu_now = hat_imu[-1, :]
                mse = (np.square(x_imu_buffer[-1] - hat_imu_now)).mean()
                mse = np.round(mse, ROUND_DIGIT)
                if mse > REJECT_MSE_THRE:
                    print(f"mse = {mse} > {REJECT_MSE_THRE}, Using old msg.")
                    reject_cnt += 1
                    new_bag.write(topic, new_imu_msg, new_imu_msg.header.stamp)
                    continue
                if mse > max_loss:
                    max_loss = mse

                new_imu_msg.angular_velocity.x = hat_imu_now[0]
                new_imu_msg.angular_velocity.y = hat_imu_now[1]
                new_imu_msg.angular_velocity.z = hat_imu_now[2]
                new_imu_msg.linear_acceleration.x = hat_imu_now[3]
                new_imu_msg.linear_acceleration.y = hat_imu_now[4]
                new_imu_msg.linear_acceleration.z = hat_imu_now[5]

            new_bag.write(topic, new_imu_msg, new_imu_msg.header.stamp)
        else:
            new_bag.write(topic, msg, msg.header.stamp)

    bag.close()
    new_bag.close()
    print("Benchmark done for {} with max loss as {}, skipping {} / {}".format(bag_fp, max_loss, reject_cnt, msg_cnt))


if __name__ == '__main__':
    fire.Fire()
