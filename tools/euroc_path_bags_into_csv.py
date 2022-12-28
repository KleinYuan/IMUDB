"""
it seems all available tools cannot do an easy job:
- plot the paths of different algorithms and display it nicely

Therefore, this scripts assume:
1. you have recorded the ground truth nav_msgs/Path as a bag
2. you have reocrded the IMUDB nav_msgs/Path as a bag
3. you have recorded the openvins baseline nav_msgs/Path as a bag
"""

import fire
import rosbag
import pandas as pd


def extract_x_y_into_csv_for_one_bag(bag_path, key, look_up_table):
    bag = rosbag.Bag(bag_path)
    for topic, msg, t in bag.read_messages():
        pose_stamped = msg.poses[-1]
        look_up_table[key].append(msg.poses[-1].pose.position.x)
        with open(bag_path + '.txt', 'a') as file:
            ts = str(pose_stamped.header.stamp.secs) + '.' + str(pose_stamped.header.stamp.nsecs)
            t_x = pose_stamped.pose.position.x
            t_y = pose_stamped.pose.position.y
            t_z = pose_stamped.pose.position.z
            q_x = pose_stamped.pose.orientation.x
            q_y = pose_stamped.pose.orientation.y
            q_z = pose_stamped.pose.orientation.z
            q_w = pose_stamped.pose.orientation.w

            file.write('{} {} {} {} {} {} {} {}\n'.format(ts, t_x, t_y, t_z, q_x, q_y, q_z, q_w))

    bag.close()
    print("Length of {} is {}".format(key, len(look_up_table[key])))
    return look_up_table


"""
python tools/euroc_path_bags_into_csv.py extract_x_y_into_csv \
 --gt_bag_path ~/EuRoC/Results/MH_04_difficult/gt.bag \
 --baseline_bag_path ~/EuRoC/Results/MH_04_difficult/open-vins.bag  \
 --imudb_bag_path ~/EuRoC/Results/MH_04_difficult/imudb.bag 
"""


def extract_x_y_into_csv(gt_bag_path, baseline_bag_path, imudb_bag_path):

    trajs_xyz = {
        'gt': [],
        'open-vins': [],
        'imudb': []
    }
    trajs_xyz = extract_x_y_into_csv_for_one_bag(gt_bag_path, 'gt', trajs_xyz)
    trajs_xyz = extract_x_y_into_csv_for_one_bag(baseline_bag_path, 'open-vins', trajs_xyz)
    trajs_xyz = extract_x_y_into_csv_for_one_bag(imudb_bag_path, 'imudb', trajs_xyz)

    df = pd.DataFrame(trajs_xyz)
    df.to_csv('gt_open_vins_imudb.csv')


if __name__ == '__main__':
    fire.Fire()
