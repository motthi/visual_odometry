from __future__ import annotations
import cv2
import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R


def create_save_directories(dir: str):
    """Create directories to save results.

    Args:
        src (str): Dataset directory.
    """
    shutil.rmtree(f"{dir}/disps/") if os.path.exists(f"{dir}/disps/") else None
    shutil.rmtree(f"{dir}/kpts/") if os.path.exists(f"{dir}/kpts/") else None
    shutil.rmtree(f"{dir}/matched_kpts/") if os.path.exists(f"{dir}/matched_kpts/") else None
    os.makedirs(f"{dir}/disps/", exist_ok=True)
    os.makedirs(f"{dir}/kpts/", exist_ok=True)
    os.makedirs(f"{dir}/matched_kpts/", exist_ok=True)


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
        l_img = cv2.imread(f"{src}/left/{i:04d}.png")
        r_img = cv2.imread(f"{src}/right/{i:04d}.png")
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


def form_transf(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1.0
    return T


def save_pose_quat(
    src: str,
    timestamps: np.ndarray, poses: np.ndarray, quats: np.ndarray,
    fmt: str = 'tum'
):
    if fmt == 'tum':
        with open(src, 'w') as f:
            for ts, p, q in zip(timestamps, poses, quats):
                f.write(f"{ts:f} {p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
    elif fmt == 'kitti':
        for pose, quat in zip(poses, quats):
            T = form_transf(R.from_quat(quat).as_matrix(), pose)
            T = T.flatten()[:12]
            f.write(f"{' '.join(map(str, T))}\n")
    else:
        raise ValueError(f"Unknown format: {fmt}")
