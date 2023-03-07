import cv2
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
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
