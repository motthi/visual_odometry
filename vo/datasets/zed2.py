import re
import numpy as np


def read_poses_quats(src: str):
    with open(src) as f:
        lines = f.readlines()
    poses = []
    quats = []
    for line in lines:
        if "AKI" in line:
            data = line.split(",")
            poses.append([float(data[5]), float(data[6]), float(data[7])])
            quats.append([float(data[8]), float(data[9]), float(data[10]), float(data[11])])
    poses = np.array(poses, dtype=np.float32)
    quats = np.array(quats, dtype=np.float32)
    return poses, quats


def read_camera_pose(src: str, step: int = 1):
    with open(src) as f:
        lines = f.readlines()
    poses = []
    quats = []
    for line in lines[::step]:
        data = line.split(",")
        poses.append([float(data[0]), float(data[1]), float(data[2])])
        quats.append([float(data[3]), float(data[4]), float(data[5]), float(data[6])])
    poses = np.array(poses, dtype=np.float32)
    quats = np.array(quats, dtype=np.float32)
    return poses, quats


def camera_params(src: str):
    # params = np.load(src)
    # left_intrinsics = params["left_intrinsic"]
    # right_intrinsics = params["right_intrinsic"]
    # left_extrinsics = params["left_extrinsic"]
    # right_extrinsics = params["right_extrinsic"]

    left_intrinsic = np.array([
        [264.677490234375, 0.0, 335.0849914550781],
        [0.0, 264.5, 186.73500061035156],
        [0.0, 0.0, 1.0]
    ])
    right_intrinsic = np.array([
        [264.677490234375, 0.0, 335.0849914550781],
        [0.0, 264.5, 186.73500061035156],
        [0.0, 0.0, 1.0]
    ])
    heading = 0.0
    depression = np.deg2rad(45)
    theta = heading - np.pi / 2
    phi = - depression - np.pi
    # phi = np.pi / 2 - depression
    # rot = np.array([
    #     [np.cos(theta), -np.sin(theta), 0.0],
    #     [np.sin(theta), np.cos(theta), 0.0],
    #     [0.0, 0.0, 1.0]
    # ]) @ np.array([
    #     [1.0, 0.0, 0.0],
    #     [0.0, np.cos(phi), -np.sin(phi)],
    #     [0.0, np.sin(phi), np.cos(phi)]
    # ])
    # rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ rot
    rot = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
    # rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]) @ rot
    trans_l = np.array([[0.0, 0.0, 0.0]]).T
    trans_r = np.array([[31.753952026367188 / right_intrinsic[0][0], 0.0, 0.0]]).T
    left_extrinsics = np.hstack((rot.T, -rot.T @ trans_l))
    right_extrinsics = np.hstack((rot.T, -rot.T @ trans_r))
    P_l = left_intrinsic @ left_extrinsics
    P_r = right_intrinsic @ right_extrinsics
    left_camera_params = {'intrinsic': left_intrinsic, 'extrinsic': left_extrinsics, 'projection': P_l}
    right_camera_params = {'intrinsic': right_intrinsic, 'extrinsic': right_extrinsics, 'projection': P_r}
    # np.savez("camera_params", left_intrinsic=intrinsic_param, left_extrinsic=extrinsic_param_l, left_projection=P_l, right_intrinsic=intrinsic_param, right_extrinsic=extrinsic_param_r, right_projection=P_r)
    return left_camera_params, right_camera_params
