import glob
import json
import numpy as np
from vo.datasets import ImageDataset


class AkiDataset(ImageDataset):
    def __init__(self, dataset_dir: str, start: int = 0, last: int = None, step: int = 1):
        super().__init__(dataset_dir, start, last, step)
        self.l_img_srcs = sorted(glob.glob(f"{self.dataset_dir}/left/*.png"))
        self.r_img_srcs = sorted(glob.glob(f"{self.dataset_dir}/right/*.png"))
        if last is None:
            last = len(self.l_img_srcs)
            self.last = len(self.l_img_srcs)
        self.l_img_srcs = self.l_img_srcs[start:last:step]
        self.r_img_srcs = self.r_img_srcs[start:last:step]

    def camera_params(self) -> list[dict]:
        with open(f"{self.dataset_dir}/camera_params.json") as f:
            data = json.load(f)
        K_l = np.array(data["left"]["intrinsic"]).reshape(3, 3)
        K_r = np.array(data["right"]["intrinsic"]).reshape(3, 3)
        E_l = np.array(data["left"]["extrinsic"]).reshape(3, 4)
        E_r = np.array(data["right"]["extrinsic"]).reshape(3, 4)
        P_l = np.array(data["left"]["projection"]).reshape(3, 4)
        P_r = np.array(data["right"]["projection"]).reshape(3, 4)
        lc_params = {'intrinsic': K_l, 'extrinsic': E_l, 'projection': P_l}
        rc_params = {'intrinsic': K_r, 'extrinsic': E_r, 'projection': P_r}
        return lc_params, rc_params

    def read_captured_poses_quats(self) -> list[np.ndarray]:
        with open(f"{self.dataset_dir}/rover_camera_pose.csv") as f:
            lines = f.readlines()
        poses = []
        quats = []
        for line in lines:
            data = line.split(",")
            poses.append([float(data[0]), float(data[1]), float(data[2])])
            quats.append([float(data[3]), float(data[4]), float(data[5]), float(data[6])])
        poses = np.array(poses, dtype=np.float32)
        quats = np.array(quats, dtype=np.float32)
        poses, quats = self.pose_quat_slice(poses, quats, self.start, self.last, self.step)
        return poses, quats

    def read_all_poses_quats(self) -> list[np.ndarray]:
        with open(f"{self.dataset_dir}/tf_data.csv") as f:
            lines = f.readlines()
        timestamps = []
        poses = []
        quats = []
        for line in lines:
            if "AKI" in line:
                data = line.split(",")
                timestamps.append(float(data[0]))
                poses.append([float(data[5]), float(data[6]), float(data[7])])
                quats.append([float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        timestamps = np.array(timestamps, dtype=np.float32)
        poses = np.array(poses, dtype=np.float32)
        quats = np.array(quats, dtype=np.float32)
        return poses, quats


# heading = 0.0
# depression = np.deg2rad(45)
# theta = heading - np.pi / 2
# phi = depression
# rot = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
# rot = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]]) @ rot
# trans_l = np.array([[0.0, 0.0, 0.0]]).T
# trans_r = np.array([[31.642799377441406 / K_r[0][0], 0.0, 0.0]]).T
# E_l = np.hstack((rot.T, -rot.T @ trans_l))
# E_r = np.hstack((rot.T, -rot.T @ trans_r))
# P_l = K_l @ E_l
# P_r = K_r @ E_r
# K_l = np.array([
#     [263.7510070800781, 0.0, 336.1105041503906],
#     [0.0, 263.7510070800781, 182.92413330078125],
#     [0.0, 0.0, 1.0]
# ])
# K_r = np.array([
#     [263.7510070800781, 0.0, 336.1105041503906],
#     [0.0, 263.7510070800781, 182.92413330078125],
#     [0.0, 0.0, 1.0]
# ])
# with open("camera_params.json", 'w') as f:
#     json.dump({'left': left_camera_params, 'right': right_camera_params}, f)
# return left_camera_params, right_camera_params
