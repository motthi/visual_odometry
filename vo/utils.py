from __future__ import annotations
import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R


def createSaveDirectories(src: str):
    """Create directories to save results.

    Args:
        src (str): Dataset directory.
    """
    os.makedirs(f"{src}disps/", exist_ok=True)
    os.makedirs(f"{src}kpts/", exist_ok=True)
    os.makedirs(f"{src}matched_kpts/", exist_ok=True)


def load_images(src: str = "./datasets", last_img_idx: int = 30, step=1) -> list[np.ndarray, np.ndarray]:
    """Load images from a dataset.

    Args:
        src (str, optional): Dataset directory. Defaults to "./datasets".
        img_num (int, optional): Last image index to be loaded. Defaults to 30.
        step (int, optional): Image index step. Defaults to 1.

    Returns:
        list[np.ndarray, np.ndarray]: Left images and right images.
    """
    l_imgs = []
    r_imgs = []
    for i in range(0, last_img_idx, step):
        l_img = cv2.imread(f"{src}left/{i:04d}.png")
        r_img = cv2.imread(f"{src}right/{i:04d}.png")
        l_imgs.append(l_img)
        r_imgs.append(r_img)
    return l_imgs, r_imgs


def load_result_poses(src: str):
    data = np.load(src)
    estimatde_poses = data['estimated']
    truth_poses = data['truth']
    img_poses = data['img_truth']
    diff = img_poses[0] - estimatde_poses[0]
    estimatde_poses += diff
    return estimatde_poses, truth_poses, img_poses
