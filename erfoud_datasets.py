import cv2
import quaternion
import yaml
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from vo.vo import *


def load_init_pose(src: str):
    datum = load_meta_data(src, 0)
    x, y, z, qw, qx, qy, qz = datum[3:10]
    rot = R.from_quat(np.array([qx, qy, qz, qw])).as_matrix()
    trans = np.array([[x, y, z]]).T
    pose = np.vstack((np.hstack((rot, trans)), np.array([[0.0, 0.0, 0.0, 1.0]])))
    return pose


def load_meta_data(dir: str, i: int):
    with open(f"{dir}nav_cam/metadata/{i:05d}.txt") as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    datum = [float(col) for col in lines[0]]
    return datum


def camera_params(src):
    with open(f"{src}nav_cam/navcam-calibration.yaml") as f:
        obj = yaml.safe_load(f)
    intrinsic_l = np.array([obj['camera_matrix_1']['data']]).reshape(3, 3)
    intrinsic_r = np.array([obj['camera_matrix_2']['data']]).reshape(3, 3)
    rot_coef = np.array([obj['rotation_matrix']['data']]).reshape(3, 3)
    trans_coef = np.array([obj['translation_coefficients']['data']]).reshape(3, 1)

    datum = load_meta_data(src, 0)
    x, y, z, qw, qx, qy, qz = datum[3:10]
    rot_rw = R.from_quat(np.array([qx, qy, qz, qw])).as_matrix()
    trans_rw = np.array([[x, y, z]]).T
    x, y, z, qw, qx, qy, qz = datum[31:38]
    trans_sr = np.array([[x, y, z]]).T
    rot_sr = R.from_quat(np.array([qx, qy, qz, qw])).as_matrix()
    rot_sw = rot_sr @ rot_rw
    rot_ws = rot_sw.T
    trans_l = np.array([[0.0, 0.0, 0.0]]).T
    trans_r = rot_coef @ trans_coef

    extrinsic_param_l = np.hstack((rot_ws, rot_ws @ trans_l))  # Warning : wrong
    extrinsic_param_r = np.hstack((rot_ws, rot_ws @ trans_r))  # Warning : wrong
    P_l = intrinsic_l @ extrinsic_param_l
    P_r = intrinsic_r @ extrinsic_param_r
    left_camera_params = {'intrinsic': intrinsic_l, 'extrinsic': extrinsic_param_l, 'projection': P_l}
    right_camera_params = {'intrinsic': intrinsic_r, 'extrinsic': extrinsic_param_r, 'projection': P_r}
    return left_camera_params, right_camera_params


def load_images(src: str, img_num: int, step: int = 1):
    l_imgs = []
    r_imgs = []
    for i in range(img_num):
        l_img = cv2.imread(f"{src}nav_cam/rectified/left/{i*step:05d}.pgm")
        r_img = cv2.imread(f"{src}nav_cam/rectified/right/{i*step:05d}.pgm")
        l_imgs.append(l_img)
        r_imgs.append(r_img)
    return l_imgs, r_imgs


def read_poses(src: str, img_num: int, step: int = 1):
    poses = []
    for i in range(img_num):
        datum = load_meta_data(src, i * step)
        x, y, z = datum[3:6]
        poses.append([x, y, z])
    return np.array(poses)


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
    # ax.set_zlim(0, 1)
    fig.savefig("vo_results.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == "__main__":
    img_len = 100
    step = 5
    base_dir = "D:/datasets/rover/erfoud/"
    lcam_params, rcam_params = camera_params(f"{base_dir}kesskess-minnie-trajectory01-2/")
    # print(lcam_params['intrinsic'])
    # print(lcam_params['extrinsic'])
    # print(lcam_params['projection'])
    # print(rcam_params['intrinsic'])
    # print(rcam_params['extrinsic'])
    # print(rcam_params['projection'])
    l_imgs, r_imgs = load_images(f"{base_dir}kesskess-minnie-trajectory01-2/", img_len, step)

    vo = StereoVisualOdometry(lcam_params, rcam_params, l_imgs, r_imgs, num_disp=100, winSize=(5, 5))

    real_poses = read_poses(f"{base_dir}kesskess-minnie-trajectory01-2/", img_len, step)
    init_pose = load_init_pose(f"{base_dir}kesskess-minnie-trajectory01-2/")
    poses = [init_pose]

    for i in tqdm(range(img_len)):
        if i < 1:
            cur_pose = init_pose
        else:
            transf = vo.estimate_pose()
            cur_pose = np.matmul(cur_pose, transf)
            poses.append(cur_pose)

    # Draw keypoints
    # for i in range(img_len - 1):
    #     vo.draw_kpts(i)

    estimated_poses = np.array([np.array(pose[0:3, 3]).T for pose in poses])
    draw_vo_results(estimated_poses, real_poses)
