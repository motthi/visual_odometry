import cv2
import quaternion
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from vo.vo import *


def read_poses_quats(src: str):
    data = np.load(src)
    poses = data['pos'][1:]
    quats = data['quat'][1:]
    return poses, quats


def plot_trajectory(src: str):
    poses, quats = read_poses_quats(src)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2])
    ax.plot(poses[0][0], poses[0][1], poses[0][2], 'o', c="r")
    ax.plot(poses[-1][0], poses[-1][1], poses[-1][2], 'x', c="r")
    ax.set_xlim(4, 8)
    ax.set_ylim(4, 8)
    ax.set_zlim(0, 1)
    plt.show()


def camera_params():
    c = 512.0
    f = 0.2
    fx = f * 1024.0 / (f * np.tan(np.pi / 6) * 2.0)
    fy = f * 1024.0 / (f * np.tan(np.pi / 6) * 2.0)
    intrinsic_param = np.array([
        [fx, 0.0, c],
        [0.0, fy, c],
        [0.0, 0.0, 1.0]
    ])
    rot = np.array([
        [0.0, -1.0, 0.0],
        [-0.5, 0.0, -np.sqrt(3.0) / 2.0],
        [np.sqrt(3) / 2, 0.0, -0.5]
    ])
    heading = 0.0
    depression = np.pi / 6
    theta = heading - np.pi / 2
    phi = - depression - np.pi / 2
    rot = (np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]) @ np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])).T
    # rot = np.array([
    #     [0.01152869, -0.99993226, -0.0015987],
    #     [-0.50104485, -0.00439316, -0.86541017],
    #     [0.86534452, 0.01077806, -0.50106156]
    # ])
    trans_l = np.array([[0.25979647, 0.12975197, 0.23186773]]).T
    trans_r = np.array([[0.25965199, -0.1302578, 0.2308808]]).T
    extrinsic_param_l = np.hstack((rot, -rot @ trans_l))
    extrinsic_param_r = np.hstack((rot, -rot @ trans_r))
    P_l = intrinsic_param @ extrinsic_param_l
    P_r = intrinsic_param @ extrinsic_param_r
    left_camera_params = {'intrinsic': intrinsic_param, 'extrinsic': extrinsic_param_l, 'projection': P_l}
    right_camera_params = {'intrinsic': intrinsic_param, 'extrinsic': extrinsic_param_r, 'projection': P_r}
    return left_camera_params, right_camera_params


def load_images(src: str = "./", img_num: int = 30):
    l_imgs = []
    r_imgs = []
    for i in range(img_num):
        l_img = cv2.imread(f"{src}left/{i:04d}.png")
        r_img = cv2.imread(f"{src}right/{i:04d}.png")
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
    # ax.set_xlim(4, 8)
    # ax.set_ylim(4, 8)
    ax.set_zlim(0, 1)
    fig.savefig("vo_results.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    # plot_trajectory("rover_pose.npz")
    data_dir = "./datasets/feature_less/"
    # data_dir = "./datasets/feature_rich/"
    lcam_params, rcam_params = camera_params()
    img_len = 29
    l_imgs, r_imgs = load_images(f"{data_dir}", img_len)

    # vo = MonocularVisualOdometry(lcam_params, l_imgs)
    vo = StereoVisualOdometry(lcam_params, rcam_params, l_imgs, r_imgs)

    real_poses, real_quats = read_poses_quats(f"{data_dir}rover_pose.npz")
    rot = R.from_quat(np.array(real_quats[0])).as_matrix()
    trans = np.array([real_poses[0]])
    init_pose = np.vstack((np.hstack((rot, trans.T)), np.array([0.0, 0.0, 0.0, 1.0])))
    poses = [init_pose]

    for i in tqdm(range(img_len)):
        if i < 1:
            cur_pose = init_pose
        else:
            transf = vo.estimate_pose()
            cur_pose = np.matmul(cur_pose, transf)
            poses.append(cur_pose)

    estimated_poses = np.array([np.array(pose[0:3, 3]).T for pose in poses])
    draw_vo_results(estimated_poses, real_poses)
