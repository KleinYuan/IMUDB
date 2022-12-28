# This script deserialize data from a ROS bag, running inference and save the data in a separate rosbags

import yaml
import fire
import rosbag
from collections import deque
import numpy as np
from copy import deepcopy
from models.imudb import Model
import torch


"""
# Below is the benchmark run for Open-VINS + IMUDB:
python experiments/ros_tumvi_imudb.py  process_a_bag_with_ckpts \
--bag_fp=/root/TUMVI/bags/dataset-outdoors2_512_16.bag \
--config_fp=logs_2080ti/tumvi_imudb/dev_tumvi_imudb_2022-08-05T00:46:11.600224-07:00/hparams.yaml \
--ckpts_fp=checkpoints_2080ti/tumvi_imudb/dev_tumvi_imudb_2022-08-05T00:46:11.600224-07:00/tumvi_imudb-epoch=501-val_denoise_loss=0.000582.ckpt
"""

"""
# Below is the benchmark run for IMUDB (w/o FIF) + Baseline [3]:
python experiments/ros_tumvi_imudb.py  process_a_bag_with_ckpts \
--bag_fp=/root/TUMVI/bags/dataset-outdoors2_512_16.bag \
--config_fp=logs_2080ti/tumvi_imudb/dev_tumvi_imudb_2022-08-08T01:20:00.881620-07:00/hparams.yaml \
--ckpts_fp=checkpoints_2080ti/tumvi_imudb/dev_tumvi_imudb_2022-08-08T01:20:00.881620-07:00/tumvi_imudb-epoch=487-val_denoise_loss=0.000004.ckpt
"""

"""
# Below is the benchmark run for LIMU-BERT + Baseline [3]:
python experiments/ros_tumvi_imudb.py  process_a_bag_with_ckpts \
--bag_fp=/root/TUMVI/bags/dataset-outdoors2_512_16.bag \
--config_fp=logs_2080ti/tumvi_imudb/dev_tumvi_imudb_2022-08-15T23:45:00.081756-07:00/hparams.yaml \
--ckpts_fp=checkpoints_2080ti/tumvi_imudb/dev_tumvi_imudb_2022-08-15T23:45:00.081756-07:00/tumvi_imudb-epoch=415-val_loss=0.005397.ckpt
"""


def process_a_bag_with_ckpts(bag_fp, ckpts_fp, config_fp, external_backend=None, get_model=False):
    UNCERTAINTY_TOLERANCE = 0.06
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
        model.eval()
        backend_mlm = model.limu_bert_mlm.forward
        backend_nsp = model.limu_bert_nsp.forward
        if get_model:
            return backend_mlm

    x_imu_buffer = deque(maxlen=int(config['inputs']['imu_sequence_number']))

    print("Reading bag {}...".format(bag_fp))
    bag = rosbag.Bag(bag_fp)
    new_bag = rosbag.Bag(f"{bag_fp}.new.imudb_pp_wo_fif_{UNCERTAINTY_TOLERANCE}.bag", 'w')
    print("Running benchmark ....")
    max_uncertainty = 0
    last_future_uncertainty = 0
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
                x_imu_infer_2 = deepcopy(x_imu_infer)
                hat_imu_now_tensor = backend_mlm(torch.tensor(x_imu_infer))
                hat_imu_next_with_hat = backend_nsp(torch.clone(hat_imu_now_tensor)).detach().numpy()
                hat_imu_now = hat_imu_now_tensor.detach().numpy()
                hat_imu_now = hat_imu_now[0]
                hat_imu_now[:, 3:] = hat_imu_now[:, 3:] * 9.8
                hat_imu_now = hat_imu_now[-1, :]
                current_uncertainty = (np.square(x_imu_buffer[-1] - hat_imu_now)).mean()
                current_uncertainty = np.round(current_uncertainty, ROUND_DIGIT)

                hat_imu_next_with_raw = backend_nsp(torch.tensor(x_imu_infer_2)).detach().numpy()
                hat_imu_next_with_raw = hat_imu_next_with_raw[0]
                hat_imu_next_with_raw[:, 3:] = hat_imu_next_with_raw[:, 3:] * 9.8
                hat_imu_next_with_raw = hat_imu_next_with_raw[0, :]

                hat_imu_next_with_hat = hat_imu_next_with_hat[0]
                hat_imu_next_with_hat[:, 3:] = hat_imu_next_with_hat[:, 3:] * 9.8
                hat_imu_next_with_hat = hat_imu_next_with_hat[0, :]

                future_uncertainty = (np.square(hat_imu_next_with_raw - hat_imu_next_with_hat)).mean()
                future_uncertainty = np.round(future_uncertainty, ROUND_DIGIT)

                uncertainty = last_future_uncertainty + current_uncertainty

                if uncertainty > UNCERTAINTY_TOLERANCE:
                    print(f"uncertainty = {last_future_uncertainty} + {current_uncertainty} =  {uncertainty} > {UNCERTAINTY_TOLERANCE}, Using old msg.")
                    reject_cnt += 1
                    new_bag.write(topic, new_imu_msg, new_imu_msg.header.stamp)
                    last_future_uncertainty = deepcopy(future_uncertainty)
                    continue
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty

                last_future_uncertainty = deepcopy(future_uncertainty)
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
    print("Benchmark done for {} with max loss as {}, skipping {} / {}".format(bag_fp, max_uncertainty, reject_cnt, msg_cnt))


if __name__ == '__main__':
    fire.Fire()
