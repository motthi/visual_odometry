import cv2
import os
import numbers
from tqdm import tqdm


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

    def load_imgs(self):
        l_imgs, r_imgs = [], []
        print("Loading images...")
        for limg_src, rimg_src in zip(tqdm(self.l_img_srcs), self.r_img_srcs):
            l_imgs.append(cv2.imread(limg_src))
            r_imgs.append(cv2.imread(rimg_src))
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
