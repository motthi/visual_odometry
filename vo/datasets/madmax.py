import glob
import re
import yaml
import warnings
import numpy as np
from scipy.spatial.transform import Rotation as R
from vo.datasets import ImageDataset

img_timestamp_re = re.compile(r'img_rect_left_(\d+).png')


class MadmaxDataset(ImageDataset):
    def __init__(self, dataset_dir: str, start: int = 0, last: int = None, step: int = 1):
        super().__init__(dataset_dir, start, last, step)
        self.l_img_srcs = sorted(glob.glob(f"{self.dataset_dir}/img_rect_left/*.png"))
        self.r_img_srcs = sorted(glob.glob(f"{self.dataset_dir}/img_rect_right/*.png"))
        if last is not None and last > len(self.l_img_srcs):
            warnings.warn(f"last={last} is larger than the number of images in the dataset ({len(self.l_img_srcs)})")
            last = None
        if last is None:
            last = len(self.l_img_srcs)
            self.last = len(self.l_img_srcs)
        self.l_img_srcs = self.l_img_srcs[start:last:step]
        self.r_img_srcs = self.r_img_srcs[start:last:step]
        self.name = "MADMAX"

    def camera_params(self) -> list[dict]:
        with open(f"{self.dataset_dir}/../calibration/camera_rect_left_info.txt") as f:
            lcam_info = yaml.load(f, Loader=yaml.FullLoader)
        with open(f"{self.dataset_dir}/../calibration/camera_rect_right_info.txt") as f:
            rcam_info = yaml.load(f, Loader=yaml.FullLoader)
        with open(f"{self.dataset_dir}/../calibration/tf__imu_to_B.csv") as f:
            tf_imu2base = f.readlines()
        with open(f"{self.dataset_dir}/../calibration/tf__imu_to_camera_left.csv") as f:
            tf_imu2lcam = f.readlines()
        with open(f"{self.dataset_dir}/../calibration/tf__camera_left_to_camera_right.csv") as f:
            tf_lcam2rcam = f.readlines()

        T_imu2base = self.tf2T(tf_imu2base)
        T_imu2lcam = self.tf2T(tf_imu2lcam)
        T_lcam2rcam = self.tf2T(tf_lcam2rcam)

        T_base2lcam = T_imu2base @ T_imu2lcam
        T_base2rcam = T_base2lcam @ T_lcam2rcam
        # T_imu2rcam = T_imu2lcam @ T_lcam2rcam

        # K_l = np.array(lcam_info['K']).reshape(3, 3)
        # K_r = np.array(rcam_info['K']).reshape(3, 3)
        K_l = np.array([lcam_info['P']]).reshape(3, 4)[:, :3]
        K_r = np.array([rcam_info['P']]).reshape(3, 4)[:, :3]
        E_l = np.linalg.inv(T_base2lcam)[:3, :]
        E_r = np.linalg.inv(T_base2rcam)[:3, :]
        P_l = K_l @ E_l
        P_r = K_r @ E_r
        P_l = np.array([lcam_info['P']]).reshape(3, 4)  # You need to align estimated trajectory if you use this
        P_r = np.array([rcam_info['P']]).reshape(3, 4)

        self.lcam_params = {
            'intrinsic': K_l,
            'extrinsic': E_l,
            'projection': P_l,
            'distortion': np.array(lcam_info['D'])
        }
        self.rcam_params = {
            'intrinsic': K_r,
            'extrinsic': E_r,
            'projection': P_r,
            'distortion': np.array(rcam_info['D'])
        }
        return self.lcam_params, self.rcam_params

    def tf2T(self, tf):
        rot = R.from_quat(list(map(float, tf[1].strip().split(',')[3:]))).as_matrix()
        trans = np.array([list(map(float, tf[1].strip().split(',')[:3]))])
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = trans
        return T

    def read_captured_poses_quats(self) -> list[np.ndarray]:
        # ts, _, _, poses = self.read_5dof_gnss()
        ts, _, _, poses = self.read_6dof_gnss_imu()

        timestamps = []
        trans = []
        quats = []
        for img_src in self.l_img_srcs:
            img_timestamp = float(img_timestamp_re.search(img_src).group(1)) * 1e-9
            idx = np.abs(ts - img_timestamp).argmin()
            pose = poses[f'{ts[idx]:.4f}']
            timestamps.append(img_timestamp)
            trans.append(pose[:3])
            quats.append(pose[3:])
        timestamps = np.array(timestamps, dtype=np.float64)
        trans = np.array(trans, dtype=np.float32)
        quats = np.array(quats, dtype=np.float32)
        return timestamps, trans, quats

    def read_all_poses_quats(self) -> list[np.ndarray]:
        # timestamps, trans, quats, _ = self.read_5dof_gnss()
        timestamps, trans, quats, _ = self.read_6dof_gnss_imu()
        s_img_scr, e_img_scr = self.l_img_srcs[0], self.l_img_srcs[-1]
        s_img_ts = float(img_timestamp_re.search(s_img_scr).group(1)) * 1e-9
        e_img_ts = float(img_timestamp_re.search(e_img_scr).group(1)) * 1e-9
        start_idx = np.abs(timestamps - s_img_ts).argmin()
        end_idx = np.abs(timestamps - e_img_ts).argmin()
        timestamps = timestamps[start_idx:end_idx + 1]
        trans = trans[start_idx:end_idx + 1]
        quats = quats[start_idx:end_idx + 1]
        return timestamps, trans, quats

    def read_5dof_gnss(self):
        ts = []
        t = []
        q = []
        poses = {}
        with open(f"{self.dataset_dir}/ground_truth/gt_5DoF_gnss.csv") as f:
            lines = f.readlines()
        for line in lines[14:]:
            data = line.split(",")
            ts.append(float(data[0]))
            t.append([float(data[3]), float(data[4]), float(data[5])])
            q.append([float(data[10]), float(data[11]), float(data[12]), float(data[9])])   # x, y, z, w
            poses[f'{data[0]}'] = [float(data[3]), float(data[4]), float(data[5]), float(data[10]), float(data[11]), float(data[12]), float(data[9])]
        ts = np.array(ts, float)
        t = np.array(t, dtype=np.float32)
        q = np.array(q, dtype=np.float32)
        return ts, t, q, poses

    def read_6dof_gnss_imu(self):
        ts = []
        t = []
        q = []
        poses = {}
        with open(f"{self.dataset_dir}/ground_truth/gt_6DoF_gnss_and_imu.csv") as f:
            lines = f.readlines()
        for line in lines[14:]:
            data = line.split(",")
            ts.append(float(data[0]))
            t.append([float(data[1]), float(data[2]), float(data[3])])
            q.append([float(data[8]), float(data[9]), float(data[10]), float(data[7])])   # x, y, z, w
            poses[f'{data[0]}'] = [float(data[1]), float(data[2]), float(data[3]), float(data[8]), float(data[9]), float(data[10]), float(data[7])]
        ts = np.array(ts, float)
        t = np.array(t, dtype=np.float32)
        q = np.array(q, dtype=np.float32)
        return ts, t, q, poses
