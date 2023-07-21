import cv2
import os
import numbers
import numpy as np
from tqdm import tqdm


def distortion_correction(img: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    nmat, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    dst = cv2.undistort(img, K, D, None, nmat)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst


class ImageDataset():
    def __init__(self, dataset_dir, start=0, last=0, step=1):
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
        self.dataset_dir = dataset_dir
        if last is not None and isinstance(last, numbers.Integral) and start > last:
            raise ValueError(f'The start index must be larger than the last index, start: {start}, last: {last}')
        self.start = start
        self.last = last
        self.step = step
        self.l_img_srcs = []
        self.r_img_srcs = []
        self.lcam_params = {}
        self.rcam_params = {}

    def load_imgs(self):
        l_imgs, r_imgs = [], []
        print("Loading images...")
        for limg_src, rimg_src in zip(tqdm(self.l_img_srcs), self.r_img_srcs):
            if 'distortion' in self.lcam_params.keys() and 'distortion' in self.rcam_params.keys():
                l_img = distortion_correction(cv2.imread(limg_src), self.lcam_params['intrinsic'], self.lcam_params['distortion'])
                r_img = distortion_correction(cv2.imread(rimg_src), self.rcam_params['intrinsic'], self.rcam_params['distortion'])
            else:
                l_img = cv2.imread(limg_src)
                r_img = cv2.imread(rimg_src)
            l_imgs.append(l_img)
            r_imgs.append(r_img)
        return l_imgs, r_imgs

    def pose_quat_slice(self, timestamps, poses, quats, start, last, step):
        timestamps = timestamps[start:last:step]
        poses = poses[start:last:step]
        quats = quats[start:last:step]
        return timestamps, poses, quats

    def camera_params(self):
        raise NotImplementedError

    def read_capture_poses_quats(self):
        raise NotImplementedError

    def read_all_poses_quats(self):
        raise NotImplementedError
