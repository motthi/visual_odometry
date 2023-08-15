import glob
import json
import numbers
import warnings
import numpy as np
from vo.datasets import ImageDataset


class AkiDataset(ImageDataset):
    def __init__(self, dataset_dir: str, start: int = 0, last: int = None, step: int = 1):
        super().__init__(dataset_dir, start, last, step)
        self.l_img_srcs = sorted(glob.glob(f"{self.dataset_dir}/left/*.png"))
        self.r_img_srcs = sorted(glob.glob(f"{self.dataset_dir}/right/*.png"))
        if last is not None and last > len(self.l_img_srcs):
            warnings.warn(f"last={last} is larger than the number of images in the dataset ({len(self.l_img_srcs)})")
            last = None
        if last is None or not isinstance(last, numbers.Integral):
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
        # P_l = np.array(data["left"]["projection"]).reshape(3, 4)
        # P_r = np.array(data["right"]["projection"]).reshape(3, 4)
        P_l = K_l @ E_l
        P_r = K_r @ E_r
        self.lc_params = {'intrinsic': K_l, 'extrinsic': E_l, 'projection': P_l}
        self.rc_params = {'intrinsic': K_r, 'extrinsic': E_r, 'projection': P_r}
        return self.lc_params, self.rc_params

    def read_captured_poses_quats(self) -> list[np.ndarray]:
        """Load the timestamps, translations and quaternions when the images were captured.

        Returns:
            list[np.ndarray]: timestamps, poses, quaternions
        """
        with open(f"{self.dataset_dir}/gt_camera_traj.csv") as f:
            lines = f.readlines()
        timestamps = []
        trans = []
        quats = []
        for line in lines:
            data = line.split(" ")
            timestamps.append(float(data[0]))
            trans.append([float(data[1]), float(data[2]), float(data[3])])
            quats.append([float(data[4]), float(data[5]), float(data[6]), float(data[7])])
        timestamps = np.array(timestamps, float)
        trans = np.array(trans, float)
        quats = np.array(quats, float)
        timestamps, trans, quats = self.slice_trans_quats(timestamps, trans, quats, self.start, self.last, self.step)
        return timestamps, trans, quats

    def read_all_poses_quats(self) -> list[np.ndarray]:
        """Load the all timestamps, translations and quaternions

        Returns:
            list[np.ndarray]: timestamps, translations, quaternions
        """
        with open(f"{self.dataset_dir}/gt_camera_traj.csv") as f:
            lines = f.readlines()
        start_ts = float(lines[self.start].split(" ")[0])
        end_ts = float(lines[self.last-1].split(" ")[0])

        with open(f"{self.dataset_dir}/gt_all_traj.csv") as f:
            lines = f.readlines()
        timestamps = []
        trans = []
        quats = []
        for line in lines:
            if "AKI" in line:
                data = line.split(",")
                if float(data[2]) * 1e-9 < start_ts or float(data[2]) * 1e-9 > end_ts:
                    continue
                timestamps.append(float(data[2]) * 1e-9)
                trans.append([float(data[5]), float(data[6]), float(data[7])])
                quats.append([float(data[8]), float(data[9]), float(data[10]), float(data[11])])
        timestamps = np.array(timestamps, float)
        trans = np.array(trans, float)
        quats = np.array(quats, float)
        return timestamps, trans, quats