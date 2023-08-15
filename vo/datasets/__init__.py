import cv2
import os
import numbers
import numpy as np
from tqdm import tqdm


def correct_distortion(img: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Correct distortion of an image.

    Args:
        img (np.ndarray): Image to correct distortion.
        K (np.ndarray): Intrinsic matrix.
        D (np.ndarray): Distortion coefficients.

    Returns:
        np.ndarray: Corrected image.
    """
    h, w = img.shape[:2]
    nmat, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    dst = cv2.undistort(img, K, D, None, nmat)
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
        self.lcam_params, self.rcam_params = self.camera_params()

    def load_imgs(self):
        l_imgs, r_imgs = [], []
        print("Loading images...")
        for i in tqdm(range(len(self.l_img_srcs))):
            l_img, r_img = self.load_img(i)
            l_imgs.append(l_img)
            r_imgs.append(r_img)
        return l_imgs, r_imgs

    def load_img(self, idx):
        if 'distortion' in self.lcam_params.keys() and 'distortion' in self.rcam_params.keys():
            l_img = correct_distortion(cv2.imread(self.l_img_srcs[idx]), self.lcam_params['intrinsic'], self.lcam_params['distortion'])
            r_img = correct_distortion(cv2.imread(self.r_img_srcs[idx]), self.rcam_params['intrinsic'], self.rcam_params['distortion'])
        else:
            l_img = cv2.imread(self.l_img_srcs[idx])
            r_img = cv2.imread(self.r_img_srcs[idx])
        return l_img, r_img

    def slice_trans_quats(self, timestamps, poses, quats, start, last, step):
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
