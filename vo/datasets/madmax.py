import glob
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
        lcam_info_src = f"{self.dataset_dir}/../calibration/camera_rect_left_info.txt"
        rcam_info_src = f"{self.dataset_dir}/../calibration/camera_rect_right_info.txt"
        tf_imu2lcam_src = f"{self.dataset_dir}/../calibration/tf__imu_to_camera_left.csv"
        tf_imu2base = f"{self.dataset_dir}/../calibration/tf__imu_to_B.csv"
        tf_lcam2rcam = f"{self.dataset_dir}/../calibration/tf__camera_left_to_camera_right.csv"
        with open(lcam_info_src) as f:
            lcam_info = yaml.load(f, Loader=yaml.FullLoader)
        with open(rcam_info_src) as f:
            rcam_info = yaml.load(f, Loader=yaml.FullLoader)
        with open(tf_imu2lcam_src) as f:
            tf_lcam2imu = f.readlines()
        with open(tf_imu2base) as f:
            tf_imu2base = f.readlines()
        with open(tf_lcam2rcam) as f:
            tf_lcam2rcam = f.readlines()

        quat_imu_to_lcam = np.array([list(map(float, tf_lcam2imu[1].strip().split(',')[3:]))])
        rot_imu2lcam = R.from_quat(quat_imu_to_lcam).as_matrix()

        quat_lcam2rcam = np.array([list(map(float, tf_lcam2rcam[1].strip().split(',')[3:]))])
        rot_lcam2rcam = R.from_quat(quat_lcam2rcam).as_matrix()

        trans_imu2lcam = np.array([list(map(float, tf_lcam2imu[1].strip().split(',')[:3]))])
        trans_imu2B = np.array([list(map(float, tf_imu2base[1].strip().split(',')[:3]))])
        trans_B2lcam = trans_imu2lcam - trans_imu2B
        trans_lcam2rcam = np.array([list(map(float, tf_lcam2rcam[1].strip().split(',')[:3]))])

        T_B2lcam = np.eye(4)
        T_B2lcam[:3, :3] = rot_imu2lcam
        T_B2lcam[:3, 3] = trans_B2lcam

        T_lcam2rcam = np.eye(4)
        T_lcam2rcam[:3, :3] = rot_lcam2rcam
        T_lcam2rcam[:3, 3] = trans_lcam2rcam

        K_l = np.array(lcam_info['K']).reshape(3, 3)
        K_r = np.array(rcam_info['K']).reshape(3, 3)
        E_l = T_B2lcam[:3, :]
        E_r = (T_B2lcam @ T_lcam2rcam)[:3, :]

        P_l = K_l @ E_l
        P_r = K_r @ E_r

        # P_l = np.array(lcam_info['P']).reshape(3, 4)
        # P_r = np.array(rcam_info['P']).reshape(3, 4)
        lc_params = {'intrinsic': K_l, 'extrinsic': E_l, 'projection': P_l}
        rc_params = {'intrinsic': K_r, 'extrinsic': E_r, 'projection': P_r}
        return lc_params, rc_params

    def read_captured_poses_quats(self) -> list[np.ndarray]:
        poses, quats = self.read_all_poses_quats()
        poses, quats = self.pose_quat_slice(poses, quats, self.start, self.last, self.step)
        return poses, quats

    def read_all_poses_quats(self) -> list[np.ndarray]:
        with open(f"{self.dataset_dir}/ground_truth/gt_5DoF_gnss.csv") as f:
            lines = f.readlines()
        poses = []
        quats = []
        for line in lines[14:]:
            data = line.split(",")
            poses.append([float(data[3]), float(data[4]), float(data[5])])
            quats.append([float(data[10]), float(data[11]), float(data[12]), float(data[9])])   # x, y, z, w
        poses = np.array(poses, dtype=np.float32)
        quats = np.array(quats, dtype=np.float32)
        return poses, quats
