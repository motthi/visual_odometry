import glob
import re
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
from vo.datasets import ImageDataset


class KatwijkBeachDataset(ImageDataset):
    def __init__(self, dataset_dir: str, start: int = 0, last: int = None, step: int = 1):
        super().__init__(dataset_dir, start, last, step)
        self.l_img_srcs = sorted(glob.glob(f"{self.dataset_dir}/LocCam/*_0.png"))
        self.r_img_srcs = sorted(glob.glob(f"{self.dataset_dir}/LocCam/*_1.png"))
        if last is None:
            last = len(self.l_img_srcs)
            self.last = len(self.l_img_srcs)
        self.l_img_srcs = self.l_img_srcs[start:last:step]
        self.r_img_srcs = self.r_img_srcs[start:last:step]

    def a_th_to_r(self, a: float, th: float) -> np.ndarray:
        a_skew = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.cos(th) * np.eye(3) + (1 - np.cos(th)) * a * a.T + np.sin(th) * a_skew

    def camera_params(self) -> list[dict]:
        t_lcam2rcam = np.array([[0.120, 0.000, 0.000]]).T
        a_lcam2rcam = np.array([[1.000, 0.000, 0.000]]).T
        th_lcam2rcam = 0.000
        R_lcam2rcam = self.a_th_to_r(a_lcam2rcam, th_lcam2rcam)
        T_lcam2rcam = np.eye(4)
        T_lcam2rcam[:3, :3] = R_lcam2rcam
        T_lcam2rcam[:3, 3] = t_lcam2rcam.reshape(3)

        t_imu2lcam = np.array([-0.382, -0.078, 0.557])
        a_imu2lcam = np.array([[-0.842, 0.337, 0.421]]).T
        th_imu2lcam = -0.589
        R_imu2lcam = self.a_th_to_r(a_imu2lcam, th_imu2lcam)
        T_imu2lcam = np.eye(4)
        T_imu2lcam[:3, :3] = R_imu2lcam
        T_imu2lcam[:3, 3] = t_imu2lcam.reshape(3)

        T_imu2rcam = T_imu2lcam @ T_lcam2rcam

        # K_l = np.array([lcam_info['P']]).reshape(3, 4)[:, :3]
        # K_r = np.array([rcam_info['P']]).reshape(3, 4)[:, :3]
        E_l = T_imu2lcam
        E_r = T_imu2rcam
        # P_l = np.array(lcam_info['P']).reshape(3, 4)
        # P_r = np.array(rcam_info['P']).reshape(3, 4)
        # D_l = np.array(lcam_info['D'])
        # D_r = np.array(rcam_info['D'])

        self.lcam_params = {
            # 'intrinsic': K_l,
            'extrinsic': E_l,
            # 'projection': P_l,
            # 'distortion': D_l
        }
        self.rcam_params = {
            # 'intrinsic': K_r,
            'extrinsic': E_r,
            # 'projection': P_r,
            # 'distortion': D_r
        }
        return self.lcam_params, self.rcam_params

    def read_captured_poses_quats(self) -> list[np.ndarray]:
        timestamps = []
        poses = []
        quats = []
        with open(f"{self.dataset_dir}/gps-utm31.txt") as f:
            lines = f.readlines()
        for line in lines[14:]:
            data = line.split(",")
            timestamps.append(float(data[0]))
            poses.append([float(data[2]), float(data[3]), float(data[4])])
            # quats.append([float(data[8]), float(data[9]), float(data[10]), float(data[7])])   # x, y, z, w

        timestamps = np.array(timestamps)
        poses = np.array(poses, dtype=np.float32)
        quats = np.array(quats, dtype=np.float32)
        return timestamps, poses, quats

    def read_all_poses_quats(self) -> list[np.ndarray]:
        timestamps = []
        poses = []
        quats = []
        with open(f"{self.dataset_dir}/gps-utm31.txt") as f:
            lines = f.readlines()
        for line in lines[14:]:
            data = line.split(",")
            timestamps.append(float(data[0]))
            poses.append([float(data[2]), float(data[3]), float(data[4])])
            # quats.append([float(data[8]), float(data[9]), float(data[10]), float(data[7])])   # x, y, z, w

        timestamps = np.array(timestamps)
        poses = np.array(poses, dtype=np.float32)
        quats = np.array(quats, dtype=np.float32)
        return timestamps, poses, quats
