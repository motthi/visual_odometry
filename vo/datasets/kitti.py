import cv2
import json
import pykitti
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from vo.datasets import ImageDataset


class KittiDataset(ImageDataset):
    def __init__(self, dataset_dir: str, seq: str, start: int = 0, last: int = None, step: int = 1):
        if last is not None:
            frame = range(start, last, step)
        else:
            self.dataset = pykitti.odometry(f"{dataset_dir}", sequence=f"{int(seq):02d}")
            last = len(self.dataset)
            frame = range(start, last, step)
        self.dataset = pykitti.odometry(f"{dataset_dir}", sequence=f"{int(seq):02d}", frames=frame)
        self.start = start
        if last is None:
            self.last = len(self.dataset)
        else:
            self.last = last
        self.step = step
        self.lcam_params, self.rcam_params = self.camera_params()
        self.l_imgs = []
        self.r_imgs = []
        self.name = "KITTI"

    def load_imgs(self):
        self.l_imgs = []
        self.r_imgs = []
        for limg, rimg in tqdm(zip(self.dataset.cam2, self.dataset.cam3), total=len(self.dataset)):
            self.l_imgs.append(cv2.cvtColor(np.array(limg), cv2.COLOR_RGB2BGR))
            self.r_imgs.append(cv2.cvtColor(np.array(rimg), cv2.COLOR_RGB2BGR))
        return self.l_imgs, self.r_imgs

    def load_img(self, idx):
        if len(self.l_imgs) == 0:
            self.load_imgs()
        return self.l_imgs[idx], self.r_imgs[idx]

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
