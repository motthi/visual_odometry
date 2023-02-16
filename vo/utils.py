from __future__ import annotations
import cv2
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R


def createSaveDirectories(dir: str):
    """Create directories to save results.

    Args:
        src (str): Dataset directory.
    """
    shutil.rmtree(f"{dir}disps/")
    shutil.rmtree(f"{dir}kpts/")
    shutil.rmtree(f"{dir}matched_kpts/")
    os.makedirs(f"{dir}disps/", exist_ok=True)
    os.makedirs(f"{dir}kpts/", exist_ok=True)
    os.makedirs(f"{dir}matched_kpts/", exist_ok=True)


def load_images(src: str = "./datasets", last_img_idx: int = 30) -> list[np.ndarray, np.ndarray]:
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
    for i in range(0, last_img_idx):
        l_img = cv2.imread(f"{src}left/{i:04d}.png")
        r_img = cv2.imread(f"{src}right/{i:04d}.png")
        l_imgs.append(l_img)
        r_imgs.append(r_img)
    return l_imgs, r_imgs


def load_result_poses(src: str):
    data = np.load(src)
    e_poses = data['estimated_poses']
    e_quats = data['estimated_quats']
    r_poses = data['real_poses']
    r_quats = data['real_quats']
    ri_poses = data['real_img_poses']
    ri_quats = data['real_img_quats']
    diff = ri_poses[0] - e_poses[0]
    e_poses += diff
    return e_poses, e_quats, r_poses, r_quats, ri_poses, ri_quats


def quaternion_mean(quats: np.ndarray):
    m = quats.T @ quats
    w, v = np.linalg.eig(m)
    return v[:, np.argmax(w)]
