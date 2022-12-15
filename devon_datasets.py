import cv2
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from vo.vo import *

num_str = "[+-]?\d+(?:\.\d+)?(?:(?:[eE][+-]?\d+)|(?:\*10\^[+-]?\d+))?"


def rotation_matrix(a: np.ndarray, r: float) -> np.ndarray:
    return a @ a.T + (1 - a @ a.T) * np.cos(r) - skey_symmetric_matrix(a) * np.sin(r)


def skey_symmetric_matrix(x: np.ndarray) -> np.ndarray:
    x = x.T[0]
    return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])


def camera_parameters():
    r = -2.8123980539
    a_cr = np.array([[-0.2660987251, -0.7639785818, 0.5878164638]]).T
    trans_l = np.array([[0, 0.0, 0.0]]).T
    trans_r = np.array([[0.24, 0.0, 0.0]]).T
    rot = rotation_matrix(a_cr, r)

    cx, cy = 463.537109375, 635.139038086
    f = 968.999694824
    w, h = 1280, 960
    fx = f * w / (f * np.tan(np.pi / 6) * 2.0)  # warning : np.pi/6 must be changed
    fy = f * h / (f * np.tan(np.pi / 6) * 2.0)  # warning : np.pi/6 must be changed
    intrinsic_param = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ])
    extrinsic_param_l = np.hstack((rot, -rot @ trans_l))
    extrinsic_param_r = np.hstack((rot, -rot @ trans_r))
    P_l = intrinsic_param @ extrinsic_param_l
    P_r = intrinsic_param @ extrinsic_param_r
    left_camera_params = {'intrinsic': intrinsic_param, 'extrinsic': extrinsic_param_l, 'projection': P_l}
    right_camera_params = {'intrinsic': intrinsic_param, 'extrinsic': extrinsic_param_r, 'projection': P_r}
    return left_camera_params, right_camera_params


def rover_init_pose():
    a = np.array([[-0.2660987251, -0.7639785818, 0.5878164638]]).T
    r = -2.8123980539
    R = rotation_matrix(a, r)
    return R, np.array([[0.0, 0.0, 0.0]]).T


def load_images(src: str = "./", img_num: int = 30, step: int = 5, seq=0):
    l_imgs = []
    r_imgs = []
    for i in range(img_num):
        l_img = cv2.imread(f"{src}sequence-{seq:02d}/color-rectified-1280x960-s00/color-rectified-left-{i*step+1:06d}.ppm")
        r_img = cv2.imread(f"{src}sequence-{seq:02d}/color-rectified-1280x960-s00/color-rectified-right-{i*step+1:06d}.ppm")
        # l_img = cv2.imread(f"{src}sequence-{seq:02d}/color-raw-1280x960-s00/color-raw-left-{i*step+1:06d}.ppm")
        # r_img = cv2.imread(f"{src}sequence-{seq:02d}/color-raw-1280x960-s00/color-raw-right-{i*step+1:06d}.ppm")
        l_imgs.append(l_img)
        r_imgs.append(r_img)
    return l_imgs, r_imgs


def draw_vo_results(estimated_poses, real_poses):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(estimated_poses[:, 0], estimated_poses[:, 1], estimated_poses[:, 2], '-o', label='Estimated', markersize=2)
    ax.plot(real_poses[:, 0], real_poses[:, 1], real_poses[:, 2], label='Real')
    ax.plot(real_poses[0][0], real_poses[0][1], real_poses[0][2], 'o', c="r")
    ax.plot(real_poses[-1][0], real_poses[-1][1], real_poses[-1][2], 'x', c="r")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(10, 10)
    # ax.set_zlim(-10, 10)
    # fig.savefig("vo_results.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def read_real_poses(src: str = "D:/datasets/rover/devon_island_rover/"):
    with open(f"{src}/rover-traverse-logs/gps-topocentric.txt") as f:
        lines = f.readlines()
    pattern = f"({num_str})\t\s*({num_str})\t\s*({num_str})\t\s*({num_str})"
    matches = [re.match(pattern, line) for line in lines]
    # idx = [int(match.group(1)) for match in matches]
    x = np.array([[float(match.group(2)) for match in matches]]).T
    y = np.array([[float(match.group(3)) for match in matches]]).T
    z = np.array([[float(match.group(4)) for match in matches]]).T
    return np.hstack((x, y, z))


if __name__ == "__main__":
    last_img_idx = 10
    step = 1
    lcam_params, rcam_params = camera_parameters()
    # print(lcam_params['intrinsic'])
    # print(lcam_params['extrinsic'])
    # print(lcam_params['projection'])
    # print(rcam_params['intrinsic'])
    # print(rcam_params['extrinsic'])
    # print(rcam_params['projection'])
    # exit(0)
    l_imgs, r_imgs = load_images("D:/datasets/rover/devon_island_rover/", last_img_idx, step=step, seq=0)
    real_poses = read_real_poses()

    vo = StereoVisualOdometry(lcam_params, rcam_params, l_imgs, r_imgs)

    rot, trans = rover_init_pose()
    init_pose = np.vstack((np.hstack((rot, trans)), np.array([0.0, 0.0, 0.0, 1.0])))

    estimated_poses = vo.estimate_all_poses(init_pose, last_img_idx, step)

    draw_vo_results(estimated_poses, real_poses[:last_img_idx])
