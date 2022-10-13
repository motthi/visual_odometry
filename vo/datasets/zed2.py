import numpy as np


def read_poses_quats(src: str):
    with open(src) as f:
        lines = f.readlines()

    poses = []
    quats = []
    for line in lines[::100]:
        if line.startswith("frame"):
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
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]) @ np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) @ rot
    rot = rot.T
    trans_l = np.array([[0.0, 0.0, 0.0]]).T
    trans_r = np.array([[-31.753952026367188 / intrinsic_param[0][0], 0.0, 0.0]]).T
    extrinsic_param_l = np.hstack((rot, -rot @ trans_l))
    extrinsic_param_r = np.hstack((rot, -rot @ trans_r))
    P_l = intrinsic_param @ extrinsic_param_l
    P_r = intrinsic_param @ extrinsic_param_r
    left_camera_params = {'intrinsic': intrinsic_param, 'extrinsic': extrinsic_param_l, 'projection': P_l}
    right_camera_params = {'intrinsic': intrinsic_param, 'extrinsic': extrinsic_param_r, 'projection': P_r}
    return left_camera_params, right_camera_params
