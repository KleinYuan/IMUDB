# This script deserialize data from a ROS bag, running inference and save the data in a separate rosbags

import yaml
import fire
import rosbag
from collections import deque
import numpy as np
from copy import deepcopy
import multiprocessing
from models.limu_bert import Model
import torch

"""
# the model_fp can be either onnx or chekpoints
python experiments/ros_euroc_limu_bert.py  process_bags \
--bag_root=/root/EuRoC/bags \
--bag_names="['MH_02_easy.bag', 'MH_04_difficult.bag', 'V2_02_medium.bag', 'V1_03_difficult.bag', 'V1_01_easy.bag']" \
--config_fp=logs_2080ti/euroc_limu_bert/dev_limu_bert_2022-02-07T23:50:30.791880-08:00/hparams.yaml \
--model_fp=checkpoints_2080ti/euroc_limu_bert/dev_limu_bert_2022-02-07T23:50:30.791880-08:00/euroc_limu_bert-epoch=4509-val_denoise_loss=0.000001.ckpt
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
        p = multiprocessing.Process(target=backend, args=(bag_fp, model_fp, config_fp, model))
        jobs.append(p)
        p.start()


"""
# IMUDB
python experiments/ros_euroc_limu_bert.py  process_a_bag_with_ckpts \
--bag_fp=/root/EuRoC/bags/V1_01_easy.bag \
--config_fp=logs_2080ti/euroc_limu_bert/dev_limu_bert_2022-02-07T23:50:30.791880-08:00/hparams.yaml \
--ckpts_fp=checkpoints_2080ti/euroc_limu_bert/dev_limu_bert_2022-02-07T23:50:30.791880-08:00/euroc_limu_bert-epoch=4509-val_denoise_loss=0.000001.ckpt
"""

"""
# LIMU-BERT
python experiments/ros_euroc_limu_bert.py  process_a_bag_with_ckpts \
--bag_fp=/root/EuRoC/bags/V1_01_easy.bag \
--config_fp=logs_2080ti/euroc_limu_bert/dev_limu_bert_2022-02-10T23:07:59.874966-08:00/hparams.yaml \
--ckpts_fp=checkpoints_2080ti/euroc_limu_bert/dev_limu_bert_2022-02-10T23:07:59.874966-08:00/euroc_limu_bert-epoch=3882-val_loss=0.001699.ckpt
"""


def process_a_bag_with_ckpts(bag_fp, ckpts_fp, config_fp, external_backend=None, get_model=False):
    REJECT_MSE_THRE = 1000 # just a very large number to invalidate the reject mechanism, i.e. FAU
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
        backend = model.limu_bert.forward
        if get_model:
            return backend

    x_imu_buffer = deque(maxlen=int(config['inputs']['imu_sequence_number']))

    print("Reading bag {}...".format(bag_fp))
    bag = rosbag.Bag(bag_fp)
    new_bag = rosbag.Bag(f"{bag_fp}.new.limu_bert.bag", 'w')
    print("Running benchmark ....")
    max_loss = 0
    stats_template = {
        'max_abs_d_ax': 0,
        'max_abs_d_ay': 0,
        'max_abs_d_az': 0,
        'max_abs_d_lx': 0,
        'max_abs_d_ly': 0,
        'max_abs_d_lz': 0,
    }
    pred_stats = deepcopy(stats_template)
    raw_stats = deepcopy(stats_template)
    last_pred_imu_msg = None
    last_raw_imu_msg = None
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
                print(f"Before {np.round(x_imu_buffer[-1], ROUND_DIGIT)} | After {np.round(hat_imu_now, ROUND_DIGIT)} | loss: {mse}")

                if last_pred_imu_msg:
                    pred_abs_d_ax = np.round(abs(last_pred_imu_msg.angular_velocity.x - hat_imu_now[0]), ROUND_DIGIT)
                    pred_abs_d_ay = np.round(abs(last_pred_imu_msg.angular_velocity.y - hat_imu_now[1]), ROUND_DIGIT)
                    pred_abs_d_az = np.round(abs(last_pred_imu_msg.angular_velocity.z - hat_imu_now[2]), ROUND_DIGIT)
                    pred_abs_d_lx = np.round(abs(last_pred_imu_msg.linear_acceleration.x - hat_imu_now[3]), ROUND_DIGIT)
                    pred_abs_d_ly = np.round(abs(last_pred_imu_msg.linear_acceleration.y - hat_imu_now[4]), ROUND_DIGIT)
                    pred_abs_d_lz = np.round(abs(last_pred_imu_msg.linear_acceleration.z - hat_imu_now[5]), ROUND_DIGIT)
                    pred_stats['max_abs_d_ax'] = max(pred_stats['max_abs_d_ax'], pred_abs_d_ax)
                    pred_stats['max_abs_d_ay'] = max(pred_stats['max_abs_d_ay'], pred_abs_d_ay)
                    pred_stats['max_abs_d_az'] = max(pred_stats['max_abs_d_az'], pred_abs_d_az)
                    pred_stats['max_abs_d_lx'] = max(pred_stats['max_abs_d_lx'], pred_abs_d_lx)
                    pred_stats['max_abs_d_ly'] = max(pred_stats['max_abs_d_ly'], pred_abs_d_ly)
                    pred_stats['max_abs_d_lz'] = max(pred_stats['max_abs_d_lz'], pred_abs_d_lz)
                if last_raw_imu_msg:
                    raw_abs_d_ax = np.round(abs(last_raw_imu_msg.angular_velocity.x - msg.angular_velocity.x), ROUND_DIGIT)
                    raw_abs_d_ay = np.round(abs(last_raw_imu_msg.angular_velocity.y - msg.angular_velocity.y), ROUND_DIGIT)
                    raw_abs_d_az = np.round(abs(last_raw_imu_msg.angular_velocity.z - msg.angular_velocity.z), ROUND_DIGIT)
                    raw_abs_d_lx = np.round(abs(last_raw_imu_msg.linear_acceleration.x - msg.linear_acceleration.x), ROUND_DIGIT)
                    raw_abs_d_ly = np.round(abs(last_raw_imu_msg.linear_acceleration.y - msg.linear_acceleration.y), ROUND_DIGIT)
                    raw_abs_d_lz = np.round(abs(last_raw_imu_msg.linear_acceleration.z - msg.linear_acceleration.z), ROUND_DIGIT)
                    raw_stats['max_abs_d_ax'] = max(raw_stats['max_abs_d_ax'], raw_abs_d_ax)
                    raw_stats['max_abs_d_ay'] = max(raw_stats['max_abs_d_ay'], raw_abs_d_ay)
                    raw_stats['max_abs_d_az'] = max(raw_stats['max_abs_d_az'], raw_abs_d_az)
                    raw_stats['max_abs_d_lx'] = max(raw_stats['max_abs_d_lx'], raw_abs_d_lx)
                    raw_stats['max_abs_d_ly'] = max(raw_stats['max_abs_d_ly'], raw_abs_d_ly)
                    raw_stats['max_abs_d_lz'] = max(raw_stats['max_abs_d_lz'], raw_abs_d_lz)
                new_imu_msg.angular_velocity.x = hat_imu_now[0]
                new_imu_msg.angular_velocity.y = hat_imu_now[1]
                new_imu_msg.angular_velocity.z = hat_imu_now[2]
                new_imu_msg.linear_acceleration.x = hat_imu_now[3]
                new_imu_msg.linear_acceleration.y = hat_imu_now[4]
                new_imu_msg.linear_acceleration.z = hat_imu_now[5]

            new_bag.write(topic, new_imu_msg, new_imu_msg.header.stamp)
            last_pred_imu_msg = deepcopy(new_imu_msg)
            last_raw_imu_msg = deepcopy(msg)
        else:
            new_bag.write(topic, msg, msg.header.stamp)

    bag.close()
    new_bag.close()
    print("Benchmark done for {} with max loss as {}, skipping {} / {}".format(bag_fp, max_loss, reject_cnt, msg_cnt))


if __name__ == '__main__':
    fire.Fire()
