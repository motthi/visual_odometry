import json
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


def read_camera_pose(src: str):
    with open(src) as f:
        lines = f.readlines()
    poses = []
    quats = []
    for line in lines:
        data = line.split(",")
        poses.append([float(data[0]), float(data[1]), float(data[2])])
        quats.append([float(data[3]), float(data[4]), float(data[5]), float(data[6])])
    poses = np.array(poses, dtype=np.float32)
    quats = np.array(quats, dtype=np.float32)
    return poses, quats


def camera_params(src: str):
    with open(src) as f:
        data = json.load(f)
    K_l = np.array(data["left"]["intrinsic"]).reshape(3, 3)
    K_r = np.array(data["right"]["intrinsic"]).reshape(3, 3)
    E_l = np.array(data["left"]["extrinsic"]).reshape(3, 4)
    E_r = np.array(data["right"]["extrinsic"]).reshape(3, 4)
    P_l = np.array(data["left"]["projection"]).reshape(3, 4)
    P_r = np.array(data["right"]["projection"]).reshape(3, 4)
    # heading = 0.0
    # depression = np.deg2rad(45)
    # theta = heading - np.pi / 2
    # phi = depression
    # rot = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    # rot = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]]) @ rot
    # trans_l = np.array([[0.0, 0.0, 0.0]]).T
    # trans_r = np.array([[31.642799377441406 / K_r[0][0], 0.0, 0.0]]).T
    # E_l = np.hstack((rot.T, -rot.T @ trans_l))
    # E_r = np.hstack((rot.T, -rot.T @ trans_r))
    # P_l = K_l @ E_l
    # P_r = K_r @ E_r
    left_camera_params = {'intrinsic': K_l, 'extrinsic': E_l, 'projection': P_l}
    right_camera_params = {'intrinsic': K_r, 'extrinsic': E_r, 'projection': P_r}
    return left_camera_params, right_camera_params
    # K_l = np.array([
    #     [263.7510070800781, 0.0, 336.1105041503906],
    #     [0.0, 263.7510070800781, 182.92413330078125],
    #     [0.0, 0.0, 1.0]
    # ])
    # K_r = np.array([
    #     [263.7510070800781, 0.0, 336.1105041503906],
    #     [0.0, 263.7510070800781, 182.92413330078125],
    #     [0.0, 0.0, 1.0]
    # ])
    # with open("camera_params.json", 'w') as f:
    #     json.dump({'left': left_camera_params, 'right': right_camera_params}, f)
    # return left_camera_params, right_camera_params
