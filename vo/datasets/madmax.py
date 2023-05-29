import glob
import re
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from vo.datasets import ImageDataset


class MadmaxDataset(ImageDataset):
    def __init__(self, dataset_dir: str, start: int = 0, last: int = None, step: int = 1):
        super().__init__(dataset_dir, start, last, step)
        self.l_img_srcs = sorted(glob.glob(f"{self.dataset_dir}/img_rect_left/*.png"))
        self.r_img_srcs = sorted(glob.glob(f"{self.dataset_dir}/img_rect_right/*.png"))
        if last is None:
            last = len(self.l_img_srcs)
            self.last = len(self.l_img_srcs)
        self.l_img_srcs = self.l_img_srcs[start:last:step]
        self.r_img_srcs = self.r_img_srcs[start:last:step]

    def camera_params(self) -> list[dict]:
        with open(f"{self.dataset_dir}/../calibration/camera_rect_left_info.txt") as f:
            lcam_info = yaml.load(f, Loader=yaml.FullLoader)
        with open(f"{self.dataset_dir}/../calibration/camera_rect_right_info.txt") as f:
            rcam_info = yaml.load(f, Loader=yaml.FullLoader)
        with open(f"{self.dataset_dir}/../calibration/tf__imu_to_camera_left.csv") as f:
            tf_lcam2imu = f.readlines()
        with open(f"{self.dataset_dir}/../calibration/tf__imu_to_B.csv") as f:
            tf_imu2base = f.readlines()
        with open(f"{self.dataset_dir}/../calibration/tf__camera_left_to_camera_right.csv") as f:
            tf_lcam2rcam = f.readlines()

        rot_imu2lcam = R.from_quat(list(map(float, tf_lcam2imu[1].strip().split(',')[3:]))).as_matrix().T
        rot_lcam2rcam = R.from_quat(list(map(float, tf_lcam2rcam[1].strip().split(',')[3:]))).as_matrix()

        trans_imu2lcam = -np.array([list(map(float, tf_lcam2imu[1].strip().split(',')[:3]))])
        trans_imu2base = np.array([list(map(float, tf_imu2base[1].strip().split(',')[:3]))])
        trans_base2lcam = trans_imu2lcam - trans_imu2base
        trans_lcam2rcam = np.array([list(map(float, tf_lcam2rcam[1].strip().split(',')[:3]))])

        T_B2lcam = np.eye(4)
        T_B2lcam[:3, :3] = rot_imu2lcam
        T_B2lcam[:3, 3] = trans_base2lcam

        T_lcam2rcam = np.eye(4)
        T_lcam2rcam[:3, :3] = rot_lcam2rcam
        T_lcam2rcam[:3, 3] = trans_lcam2rcam

        K_l = np.array(lcam_info['P']).reshape(3, 4)[:3, :3]
        K_r = np.array(rcam_info['P']).reshape(3, 4)[:3, :3]
        # K_l = np.array(lcam_info['K']).reshape(3, 3)
        # K_r = np.array(rcam_info['K']).reshape(3, 3)
        E_l = T_B2lcam[:3, :]
        E_r = (T_lcam2rcam @ T_B2lcam)[:3, :]

        P_l = K_l @ E_l
        P_r = K_r @ E_r

        lc_params = {'intrinsic': K_l, 'extrinsic': E_l, 'projection': P_l}
        rc_params = {'intrinsic': K_r, 'extrinsic': E_r, 'projection': P_r}
        return lc_params, rc_params

    def read_captured_poses_quats(self) -> list[np.ndarray]:
        poses_data = {}
        ts = []
        with open(f"{self.dataset_dir}/ground_truth/gt_6DoF_gnss_and_imu.csv") as f:
            lines = f.readlines()
        for line in lines[14:]:
            data = line.split(",")
            poses_data[f'{data[0]}'] = [float(data[1]), float(data[2]), float(data[3]), float(data[8]), float(data[9]), float(data[10]), float(data[7])]
            ts.append(float(data[0]))

        timestamps = []
        poses = []
        quats = []
        img_timestamp_re = re.compile(r'img_rect_left_(\d+).png')
        for img_src in self.l_img_srcs:
            img_timestamp = float(img_timestamp_re.search(img_src).group(1)) * 1e-9
            idx = np.abs(np.asarray(ts) - img_timestamp).argmin()
            closest_timestamp = ts[idx]
            pose = poses_data[f'{closest_timestamp:.4f}']
            timestamps.append(img_timestamp)
            poses.append(pose[:3])
            quats.append(pose[3:])
        poses = np.array(poses, dtype=np.float32)
        quats = np.array(quats, dtype=np.float32)
        return timestamps, poses, quats

    def read_all_poses_quats(self) -> list[np.ndarray]:
        timestamps = []
        poses = []
        quats = []
        with open(f"{self.dataset_dir}/ground_truth/gt_6DoF_gnss_and_imu.csv") as f:
            lines = f.readlines()
        for line in lines[14:]:
            data = line.split(",")
            timestamps.append(float(data[0]))
            poses.append([float(data[1]), float(data[2]), float(data[3])])
            quats.append([float(data[8]), float(data[9]), float(data[10]), float(data[7])])   # x, y, z, w

        timestamps = np.array(timestamps)
        poses = np.array(poses, dtype=np.float32)
        quats = np.array(quats, dtype=np.float32)
        return timestamps, poses, quats
