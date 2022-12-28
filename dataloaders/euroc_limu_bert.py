from common.euroc_utils import pdump, pload, bmtm
from common.lie_algebra import SO3
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch
import cv2
from torchvision import transforms
from copy import deepcopy
from common.mask import create_mask


class EUROCDataset(Dataset):

    def __init__(self, config, mode):
        super().__init__()

        self.data_config = config["data"]
        self.model_config = config["model"]

        self.caches_dir = self.data_config["caches_dir"]
        self.data_dir = self.data_config["data_dir"]

        self.mode = mode  # train, val or test
        self.batch_size = self.data_config["batch_size"]
        # choose between training, validation or test sequences
        self.sequences = self.get_sequences(self.data_config["train_seqs"], self.data_config["val_seqs"], self.data_config["test_seqs"])
        self._train = False
        self._val = False
        self.read_data(self.data_dir)
        self.imu_cam_frequency_ratio = int(config['sensors']['imu']['frequency']) / int(config['sensors']['cam0']['frequency'])
        # The datasets consists of a few different datasets. To construct the datasets __getitem__, we save the
        # upper bound of the index for each sub-datasets in a dictionary to fast search
        self.index_bounds = {-1: 0}
        self.caches = self.load_caches()
        self.image_caches = {}

    def get_sequences(self, train_seqs, val_seqs, test_seqs):
        """Choose sequence list depending on dataset mode"""
        sequences_dict = {
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
        }
        return sequences_dict[self.mode]

    def __getitem__(self, absolute_index):
        # print("index is : {}".format(absolute_index))
        seq_number = None
        for k, v in self.index_bounds.items():
            if absolute_index < v:
                seq_number = k
                #print(f"{absolute_index} is smalller than {v}, namely, seq_numebr is {seq_number} with index bound: {self.index_bounds}")
                break
        if seq_number is None:
            raise Exception(f"{absolute_index} is too large so that we cannot find a sub-datasets it belongs to {self.index_bounds}")
        relative_index = absolute_index - self.index_bounds[seq_number - 1]
        sequence_data = self.caches[seq_number]['data']

        last_index = relative_index + int(self.model_config['inputs']['imu_sequence_number'])
        last_index = min(sequence_data['imu_data'].shape[0], int(sequence_data['cam0_ts'].shape[0] * self.imu_cam_frequency_ratio), last_index)

        # avoid the index go beyonds the limit for both imu and cam
        output_imu_start_index = last_index - int(self.model_config['outputs']['imu_sequence_number'])
        input_imu_start_index = last_index - int(self.model_config['inputs']['imu_sequence_number'])

        input_imu = sequence_data['imu_data'][input_imu_start_index: last_index] # (S, 6)
        normed_input_imu = deepcopy(input_imu.detach().numpy())
        normed_input_imu[:, 3:] = normed_input_imu[:, 3:] / 9.8
        # Apply Mask here
        mask_seq, masked_pos, seq = create_mask(
            normed_input_imu,
            mask_ratio=float(self.data_config['mask']['mask_ratio']),
            max_gram=float(self.data_config['mask']['max_gram']),
            mask_prob=float(self.data_config['mask']['mask_prob']),
            replace_prob=float(self.data_config['mask']['replace_prob'])
        )

        data_return = {
            'inputs': {
                'mask_seqs': torch.from_numpy(mask_seq),
                'masked_pos': torch.from_numpy(masked_pos).long(),
            },
            'outputs': {
                'seq': torch.from_numpy(seq),
                'normed_imu_seq': torch.from_numpy(normed_input_imu),
            },
            'gts': {
                'position': sequence_data['p_gt_data'][output_imu_start_index: last_index],
                'velocity': sequence_data['v_gt_data'][output_imu_start_index: last_index]
            },
            'seq_number': seq_number
        }

        return data_return

    def __len__(self):
        # print("Total data length is based on the number of IMU data: {}".format(self.caches['count']))
        # Total data length is based on the number of IMU data: 146967
        return self.caches['count']

    def load_cache(self, i):
        return pload(self.caches_dir, self.sequences[i] + '_cache.p')

    def load_caches(self):
        caches = {
            'count': 0
        }
        for i, seq in enumerate(self.sequences):
            data = pload(self.caches_dir, self.sequences[i] + '_cache.p')
            caches[i] = {
                'data': data,
                'count': len(data['imu_data'])
            }
            caches['count'] += caches[i]['count']
            self.index_bounds[i] = self.index_bounds[i-1] + caches[i]['count']
        return caches

    def read_data(self, data_dir):
        r"""Read the data from the dataset"""

        f = os.path.join(self.caches_dir, 'MH_01_easy_cache.p')
        if True and os.path.exists(f):
            return

        print("Start read_data, be patient please")

        def set_path(seq):
            path_imu = os.path.join(data_dir, seq, "mav0", "imu0", "data.csv")
            path_cam0 = os.path.join(data_dir, seq, "mav0", "cam0", "data.csv")
            path_gt = os.path.join(data_dir, seq, "mav0", "state_groundtruth_estimate0", "data.csv")
            return path_imu, path_cam0, path_gt

        sequences = os.listdir(data_dir)
        # read each sequence
        for sequence in sequences:
            print("\nSequence name: " + sequence)
            path_imu, path_cam0, path_gt = set_path(sequence)
            imu = np.genfromtxt(path_imu, delimiter=",", skip_header=1) # timestamp, wx, wy, wz, ax, ay, az
            cam0 = np.genfromtxt(path_cam0, delimiter=",", skip_header=1)
            gt = np.genfromtxt(path_gt, delimiter=",", skip_header=1)
            # the camera csv first and secon column is the same except the second one has .png
            # there is an issue to parse a csv with (float, str) into np object
            # as a workaround, we just use the fist column
            cam0[:, 1] = cam0[:, 0]
            # time synchronization between IMU and ground truth
            t0 = np.max([gt[0, 0], cam0[0, 0], imu[0, 0]])
            t_end = np.min([gt[-1, 0], cam0[-1, 0], imu[-1, 0]])

            # start index
            idx0_imu = np.searchsorted(imu[:, 0], t0)
            idx0_cam0 = np.searchsorted(cam0[:, 0], t0)
            idx0_gt = np.searchsorted(gt[:, 0], t0)

            # end index
            idx_end_imu = np.searchsorted(imu[:, 0], t_end, 'right')
            idx_end_gt = np.searchsorted(gt[:, 0], t_end, 'right')
            idx_end_cam0 = np.searchsorted(cam0[:, 0], t_end, 'right')

            # subsample
            imu = imu[idx0_imu: idx_end_imu]
            gt = gt[idx0_gt: idx_end_gt]
            cam0 = cam0[idx0_cam0: idx_end_cam0]
            # Be aware that the camera is running at 20Hz while IMU is at 200 Hz
            # Namely, there will be 100 times more IMU data than camera
            print(f" imu started from {idx0_imu} to {idx_end_imu}, {len(imu)} in total")
            print(f" cam0 started from {idx0_cam0} to {idx_end_cam0}, {len(cam0)} in total")

            ts = imu[:, 0] / 1e9

            # # interpolate
            gt = self.interpolate(gt, gt[:, 0] / 1e9, ts)
            #
            # # take ground truth position
            p_gt = gt[:, 1:4]  # xyz
            p_gt = p_gt - p_gt[0]

            v_gt = gt[:, 8:11]
            v_gt = torch.Tensor(v_gt).double()

            # # convert from numpy
            p_gt = torch.Tensor(p_gt).double()
            imu = torch.Tensor(imu[:, 1:]).double()

            mondict = {
                'imu_data': imu.float(),
                'p_gt_data': p_gt.float(),
                'v_gt_data': v_gt.float(),
                'cam0_ts': cam0[:, 1]
            }
            pdump(mondict, self.caches_dir, sequence + "_cache.p")

    @staticmethod
    def interpolate(x, t, t_int):
            """
            Interpolate ground truth at the sensor timestamps
            """

            # vector interpolation
            x_int = np.zeros((t_int.shape[0], x.shape[1]))
            for i in range(x.shape[1]):
                if i in [4, 5, 6, 7]:
                    continue
                x_int[:, i] = np.interp(t_int, t, x[:, i])
            # quaternion interpolation
            t_int = torch.Tensor(t_int - t[0])
            t = torch.Tensor(t - t[0])
            qs = SO3.qnorm(torch.Tensor(x[:, 4:8]))
            x_int[:, 4:8] = SO3.qinterp(qs, t, t_int).numpy()
            return x_int
