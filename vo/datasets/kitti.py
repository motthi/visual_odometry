import cv2
import json
import glob
import numbers
import pykitti
import warnings
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from vo.datasets import ImageDataset


class KittiDataset(ImageDataset):
    def __init__(self, dataset_dir: str, seq: int, start: int = 0, last: int = None, step: int = 1):
        self.l_img_srcs = sorted(glob.glob(f"{dataset_dir}/sequences/{seq:02d}/image_2/*.png"))
        self.r_img_srcs = sorted(glob.glob(f"{dataset_dir}/sequences/{seq:02d}/image_3/*.png"))

        if start < 0:
            warnings.warn(f"start={start} is negative")
            start = 0
        if start > len(self.l_img_srcs):
            warnings.warn(f"start={start} is larger than the number of images in the dataset ({len(self.l_img_srcs)})")
            start = 0
        if last is not None and last > len(self.l_img_srcs):
            warnings.warn(f"last={last} is larger than the number of images in the dataset ({len(self.l_img_srcs)})")
            last = None
        if last is None or not isinstance(last, numbers.Integral):
            last = len(self.l_img_srcs)
        self.start = start
        self.step = step
        self.last = last

        self.dataset = pykitti.odometry(f"{dataset_dir}", sequence=f"{seq:02d}", frames=range(start, last, step))
        self.step = step
        self.lcam_params, self.rcam_params = self.camera_params()
        self.l_img_srcs = self.l_img_srcs[start:last:step]
        self.r_img_srcs = self.r_img_srcs[start:last:step]
        self.load_img_info()
        self.name = "KITTI"

    def read_captured_poses_quats(self):
        ts, t, r = self.read_all_poses_quats()
        return ts[self.start:self.last:self.step], t[self.start:self.last:self.step], r[self.start:self.last:self.step]

    def read_all_poses_quats(self) -> list[np.ndarray]:
        timestamps = []
        trans = []
        quats = []
        timestamps = [t.total_seconds() for t in self.dataset.timestamps]
        poses = self.dataset.poses
        for pose in poses:
            trans.append(pose[:3, 3])
            rot = pose[:3, :3]
            quat = R.from_matrix(rot).as_quat()
            quats.append(quat)
        return timestamps, trans, quats

    def camera_params(self) -> list[dict]:
        self.lcam_params = {
            'intrinsic': self.dataset.calib.K_cam2,
            'extrinsic': self.dataset.calib.T_cam2_velo[:3, :],
            'projection': self.dataset.calib.P_rect_20
        }
        self.rcam_params = {
            'intrinsic': self.dataset.calib.K_cam3,
            'extrinsic': self.dataset.calib.T_cam3_velo[:3, :],
            'projection': self.dataset.calib.P_rect_30
        }
        return self.lcam_params, self.rcam_params

    def save_info(self, src):
        data = {
            'start': self.start,
            'last': self.last,
            'step': self.step,
            'camera_params': {
                'lcam': {
                    'intrinsic': self.lcam_params['intrinsic'].tolist(),
                    'extrinsic': self.lcam_params['extrinsic'].tolist(),
                    'projection': self.lcam_params['projection'].tolist()
                },
                'rcam': {
                    'intrinsic': self.rcam_params['intrinsic'].tolist(),
                    'extrinsic': self.rcam_params['extrinsic'].tolist(),
                    'projection': self.rcam_params['projection'].tolist()
                }
            }
        }
        if 'distortion' in self.lcam_params.keys() and 'distortion' in self.rcam_params.keys():
            data['camera_params']['lcam']['distortion'] = self.lcam_params['distortion'].tolist()
            data['camera_params']['rcam']['distortion'] = self.rcam_params['distortion'].tolist()
        with open(f"{src}", 'w') as f:
            json.dump(data, f, indent=4)
