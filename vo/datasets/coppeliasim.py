import numpy as np

def read_poses_quats(src: str):
    data = np.load(src)
    poses = data['pos'][1:]
    quats = data['quat'][1:]
    img_poses = data['image_pos']
    img_quats = data['image_pos']
    return poses, quats, img_poses, img_quats


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
    heading = 0.0
    depression = np.pi / 6
    theta = heading - np.pi / 2
    phi = - depression - np.pi / 2
    rot = (np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]]) @ np.array([[1.0, 0.0, 0.0], [0.0, np.cos(phi), -np.sin(phi)], [0.0, np.sin(phi), np.cos(phi)]])).T
    trans_l = np.array([[0.25971, 0.059796, 0.73315]]).T
    trans_r = np.array([[0.25974, -0.059997, 0.73329]]).T
    extrinsic_param_l = np.hstack((rot, -rot @ trans_l))
    extrinsic_param_r = np.hstack((rot, -rot @ trans_r))
    P_l = intrinsic_param @ extrinsic_param_l
    P_r = intrinsic_param @ extrinsic_param_r
    left_camera_params = {'intrinsic': intrinsic_param, 'extrinsic': extrinsic_param_l, 'projection': P_l}
    right_camera_params = {'intrinsic': intrinsic_param, 'extrinsic': extrinsic_param_r, 'projection': P_r}
    return left_camera_params, right_camera_params

