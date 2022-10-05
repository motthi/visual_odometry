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
    intrinsic_param = np.array([
        [264.677490234375, 0.0, 335.0849914550781],
        [0.0, 264.5, 186.73500061035156],
        [0.0, 0.0, 1.0]
    ])

    heading = 0.0
    depression = np.pi / 4
    theta = heading - np.pi / 2
    phi = - depression - np.pi / 2
    rot = (np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]) @ np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])).T
    trans_l = np.array([[0.0, 0.0, 0.0]]).T
    # trans_r = np.array([[-31.753952026367188, 0.0, 0.0]]).T   # fx * B
    trans_r = np.array([[-31.753952026367188 / intrinsic_param[0][0], 0.0, 0.0]]).T
    extrinsic_param_l = np.hstack((rot, -rot @ trans_l))
    extrinsic_param_r = np.hstack((rot, -rot @ trans_r))
    P_l = intrinsic_param @ extrinsic_param_l
    P_r = intrinsic_param @ extrinsic_param_r
    left_camera_params = {'intrinsic': intrinsic_param, 'extrinsic': extrinsic_param_l, 'projection': P_l}
    right_camera_params = {'intrinsic': intrinsic_param, 'extrinsic': extrinsic_param_r, 'projection': P_r}
    return left_camera_params, right_camera_params


def load_images(src: str = "./", img_num: int = 30, step=1):
    l_imgs = []
    r_imgs = []
    for i in range(img_num):
        l_img = cv2.imread(f"{src}left/{i*step+1:04d}.png")
        r_img = cv2.imread(f"{src}right/{i*step+1:04d}.png")
        l_imgs.append(l_img)
        r_imgs.append(r_img)
    return l_imgs, r_imgs


def draw_vo_results(estimated_poses, save_src: str = None):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(estimated_poses[:, 0], estimated_poses[:, 1], estimated_poses[:, 2], '-o', label='Estimated', markersize=2)
    # ax.plot(real_poses[:, 0], real_poses[:, 1], real_poses[:, 2], label='Real')
    # ax.plot(real_poses[0][0], real_poses[0][1], real_poses[0][2], 'o', c="r")
    # ax.plot(real_poses[-1][0], real_poses[-1][1], real_poses[-1][2], 'x', c="r")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_xlim(4, 8)
    # ax.set_ylim(4, 8)
    # ax.set_zlim(-1, 1)
    fig.savefig(save_src, dpi=300, bbox_inches='tight', pad_inches=0) if save_src is not None else None
    plt.show()


if __name__ == "__main__":
    data_dir = "./datasets/20220929/"
    lcam_params, rcam_params = camera_params()
    step = 1
    img_len = int(325 / step)
    l_imgs, r_imgs = load_images(f"{data_dir}", img_len, step)

    vo = StereoVisualOdometry(lcam_params, rcam_params, l_imgs, r_imgs, num_disp=50)

    # real_poses, real_quats = read_poses_quats(f"{data_dir}rover_pose.npz")
    # rot = R.from_quat(np.array(real_quats[0])).as_matrix()
    # trans = np.array([real_poses[0]])
    rot = np.eye(3)
    trans = np.array([[0, 0, 0]])
    init_pose = np.vstack((np.hstack((rot, trans.T)), np.array([0.0, 0.0, 0.0, 1.0])))
    poses = [init_pose]

    for i in tqdm(range(img_len)):
        if i < 1:
            cur_pose = init_pose
        else:
            transf = vo.estimate_pose()
            if transf is None:
                continue
            cur_pose = np.matmul(cur_pose, transf)
            poses.append(cur_pose)

    estimated_poses = np.array([np.array(pose[0:3, 3]).T for pose in poses])
    np.savez(f"{data_dir}vo_result_poses.npz", estimated_poses=estimated_poses)
    draw_vo_results(estimated_poses, f"{data_dir}vo_result.png")

    # Draw keypoints
    for i in tqdm(range(img_len - 1)):
        vo.draw_kpts(i, f"{data_dir}kpts/")
    # Draw disparities
    for i in tqdm(range(img_len - 1)):
        vo.draw_disp(i, f"{data_dir}disps/")
