import cv2
import os


class ImageDataset():
    def __init__(self, dataset_dir, start=0, last=0, step=1):
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")
        self.dataset_dir = dataset_dir
        self.start = start
        self.last = last
        self.step = step
        self.l_img_srcs = []
        self.r_img_srcs = []

    def load_imgs(self):
        l_imgs, r_imgs = [], []
        for limg_src, rimg_src in zip(self.l_img_srcs, self.r_img_srcs):
            l_imgs.append(cv2.imread(limg_src))
            r_imgs.append(cv2.imread(rimg_src))
        return l_imgs, r_imgs

    def pose_quat_slice(self, poses, quats, start, last, step):
        poses = poses[start:last:step]
        quats = quats[start:last:step]
        return poses, quats

    def camera_params(self):
        raise NotImplementedError

    def read_capture_poses_quats(self):
        raise NotImplementedError

    def read_all_poses_quats(self):
        raise NotImplementedError
