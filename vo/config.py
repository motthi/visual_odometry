import cv2
import json
import numpy as np
from vo.detector import HarrisCornerDetector
from vo.tracker import BruteForceTracker


class ConfigLoader():
    def __init__(self) -> None:
        self.config = None

    @property
    def data(self):
        return self.config

    def set(self):
        raise NotImplementedError

    def save(self, src):
        with open(src, "w") as f:
            json.dump(self.config, f)


class AkiConfigLoader(ConfigLoader):
    def __init__(self):
        super().__init__()

    def set(self, l_img: np.ndarray):
        # Feature detector
        detector = HarrisCornerDetector(blocksize=5, ksize=5, thd=0.01)

        # Feature descriptor
        descriptor = cv2.ORB_create()

        # Tracker
        max_track_dist = 50
        tracker = BruteForceTracker(max_track_dist, cv2.NORM_HAMMING, cross_check=True)

        D = 50
        img_mask = np.full(l_img.shape[:2], 255, dtype=np.uint8)
        img_mask[:D, :] = 0     # Y min
        img_mask[-100:, :] = 0  # Y max
        img_mask[:, :D] = 0     # X min
        img_mask[:, -D:] = 0    # X max

        max_iter = 84
        inlier_thd = 0.01

        max_disp = 50
        use_disp = False

        self.config = {
            "detector": detector,
            "descriptor": descriptor,
            "tracker": tracker,
            "max_track_dist": max_track_dist,
            "img_mask": img_mask,
            "max_iter": max_iter,
            "inlier_thd": inlier_thd,
            "max_disp": max_disp,
            "use_disp": use_disp
        }

    def load(self):
        return self.config


class MadmaxConfigLoader(ConfigLoader):
    def __init__(self):
        super().__init__()

    def set(self, l_img: np.ndarray):
        # Feature detector
        detector = cv2.ORB_create(nfeatures=3000)

        # Feature descriptor
        descriptor = cv2.ORB_create(nfeatures=3000)

        # Tracker
        max_track_dist = 100
        tracker = BruteForceTracker(max_track_dist, cv2.NORM_HAMMING, cross_check=True)

        D = 50
        img_mask = np.full(l_img.shape[:2], 255, dtype=np.uint8)
        img_mask[:D, :] = 0     # Y min
        img_mask[-100:, :] = 0  # Y max
        img_mask[:, :D] = 0     # X min
        img_mask[:, -D:] = 0    # X max

        max_iter = 300
        inlier_thd = 0.05

        max_disp = 80
        use_disp = False

        self.config = {
            "detector": detector,
            "descriptor": descriptor,
            "tracker": tracker,
            "max_track_dist": max_track_dist,
            "img_mask": img_mask,
            "max_iter": max_iter,
            "inlier_thd": inlier_thd,
            "max_disp": max_disp,
            "use_disp": use_disp
        }
