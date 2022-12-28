import yaml
import fire
import rosbag
from collections import deque
import numpy as np
from models.imudb import Model
import torch


"""
# Below is the benchmark run in the slides:
python experiments/profile_latency.py  profile_with_a_bag_and_ckpts \
--bag_fp=/root/EuRoC/bags/MH_04_difficult.bag \
--config_fp=logs_1080tis/euroc_imudb/dev_imudb_2022-02-20T00:29:37.077961-08:00/hparams.yaml \
--ckpts_fp=checkpoints_1080tis/euroc_imudb/dev_imudb_2022-02-20T00:29:37.077961-08:00/euroc_imudb-epoch=3100-val_denoise_loss=0.000006.ckpt
"""


def profile_with_a_bag_and_ckpts(bag_fp, ckpts_fp, config_fp, external_backend=None, get_model=False):
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device} device")
        print("Initializing the ckpts_fp session ...")
        model = Model.load_from_checkpoint(ckpts_fp)
        model.eval()
        backend_mlm = model.limu_bert_mlm.forward
        if get_model:
            return backend_mlm

    x_imu_buffer = deque(maxlen=int(config['inputs']['imu_sequence_number']))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    print("Reading bag {}...".format(bag_fp))
    bag = rosbag.Bag(bag_fp)
    summed_curr_time = 0
    for topic, msg, t in bag.read_messages():
        if topic == '/imu0':
            msg_cnt += 1
            # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Imu.html
            x_imu_buffer.append(
                [
                    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                    msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
                ]
            ) # (S, 6)
            buffer_is_valid = (len(x_imu_buffer) == int(config['inputs']['imu_sequence_number']))

            if buffer_is_valid:

                x_imu_infer = np.array([x_imu_buffer]).astype(np.float32)
                x_imu_infer[:, :, 3:] = x_imu_infer[:, :, 3:] / 9.8
                starter.record()
                hat_imu_now_tensor = backend_mlm(torch.tensor(x_imu_infer))
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                summed_curr_time = summed_curr_time + curr_time
    print(summed_curr_time / msg_cnt)


if __name__ == '__main__':
    fire.Fire()
