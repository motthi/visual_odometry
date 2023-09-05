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


def load_result_poses(src: str):
    data = np.load(src)
    est_timestamps = data['est_timestamps']
    est_poses = data['est_poses']
    est_quats = data['est_quats']
    gt_all_timestamps = data['gt_timestamps']
    gt_all_poses = data['gt_poses']
    gt_all_quats = data['gt_quats']
    gt_timestamps = data['gt_img_timestamps']
    gt_poses = data['gt_img_poses']
    gt_quats = data['gt_img_quats']

    est_ps, gt_all_ps, gt_ps = [], [], []
    for est_p, e_q in zip(est_poses, est_quats):
        rot = R.from_quat(e_q).as_matrix()
        T_est = form_transf(rot, est_p)
        est_ps.append(T_est)
    est_ps = np.array(est_ps)

    for gt_all_p, gt_all_q in zip(gt_all_poses, gt_all_quats):
        rot = R.from_quat(gt_all_q).as_matrix()
        T_gt_all = form_transf(rot, gt_all_p)
        gt_all_ps.append(T_gt_all)
    gt_all_ps = np.array(gt_all_ps)

    for gt_p, gt_q in zip(gt_poses, gt_quats):
        rot = R.from_quat(gt_q).as_matrix()
        T_gt = form_transf(rot, gt_p)
        gt_ps.append(T_gt)
    gt_ps = np.array(gt_ps)
    return est_timestamps, est_ps, gt_all_timestamps, gt_all_ps, gt_timestamps, gt_ps


def quaternion_mean(quats: np.ndarray):
    m = quats.T @ quats
    w, v = np.linalg.eig(m)
    return v[:, np.argmax(w)]


def form_transf(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1.0
    return T


def trans_quats_to_poses(quats: np.ndarray, trans: np.ndarray) -> np.ndarray:
    poses = []
    for t, q in zip(trans, quats):
        rot = R.from_quat(q).as_matrix()
        pose = form_transf(rot, t)
        poses.append(pose)
    return np.array(poses)


def poses_to_trans_quats(poses: np.ndarray) -> np.ndarray:
    trans = []
    quats = []
    for pose in poses:
        tran = pose[:3, 3]
        rot = pose[:3, :3]
        quat = R.from_matrix(rot).as_quat()
        trans.append(tran)
        quats.append(quat)
    trans = np.array(trans)
    quats = np.array(quats)
    return trans, quats


def save_trajectory(
    src: str,
    timestamps: np.ndarray, poses: np.ndarray, quats: np.ndarray,
    fmt: str = 'tum'
) -> None:
    if fmt == 'tum':
        with open(src, 'w') as f:
            for ts, p, q in zip(timestamps, poses, quats):
                f.write(f"{ts:f} {p[0]} {p[1]} {p[2]} {q[0]} {q[1]} {q[2]} {q[3]}\n")
    elif fmt == 'kitti':
        with open(src, 'w') as f:
            for pose, quat in zip(poses, quats):
                T = form_transf(R.from_quat(quat).as_matrix(), pose)
                T = T.flatten()[:12]
                f.write(f"{' '.join(map(str, T))}\n")
    else:
        raise ValueError(f"Unknown format: {fmt}")


def trajectory_length(poses: np.ndarray) -> float:
    return np.sum(np.linalg.norm(poses[1:, :3, 3] - poses[:-1, :3, 3], axis=1))


def umeyama_alignment(x: np.ndarray, y: np.ndarray, with_scale: bool = False, align_start: bool = False) -> tuple:
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        raise Exception("data matrices must have the same shape")

    dim, n = x.shape  # m = dimension, n = nr. of data points

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((dim, dim))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)
    if np.count_nonzero(d > np.finfo(d.dtype).eps) < dim - 1:
        raise Exception("Degenerate covariance rank, Umeyama alignment is not possible")

    # S matrix, eq. 43
    s = np.eye(dim)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[dim - 1, dim - 1] = -1

    # rotation, eq. 40
    rot = u @ s @ v

    # scale & translation, eq. 42 and 41
    scale = 1 / sigma_x * np.trace(np.diag(d) @ s) if with_scale else 1.0

    if align_start:
        trans = y[:, 0] - scale * (rot @ x[:, 0])
    else:
        trans = mean_y - scale * (rot @ mean_x)

    return rot, trans, scale


# def transform_poses(traj, rot, trans):
#     aligned_traj = np.zeros_like(traj)
#     for i in range(len(traj)):
#         aligned_traj[i] = rot @ traj[i] + trans
#     return aligned_traj

def transform_poses(poses, rot, trans):
    ts = [pose[:3, 3] for pose in poses]
    rots = [pose[:3, :3] for pose in poses]
    aligned_trans = np.zeros_like(ts)
    for i in range(len(ts)):
        aligned_trans[i] = rot @ ts[i] + trans

    rel_rot = []
    for i in range(len(rots) - 1):
        rel_rot.append(rots[i + 1].T @ rots[i])
    rel_rot = np.array(rel_rot)

    aligned_rots = []
    aligned_rots.append(rots[0])
    for i in range(len(rel_rot)):
        aligned_rots.append(rel_rot[i] @ aligned_rots[-1])
    aligned_rots = np.array(aligned_rots)

    aligned_poses = np.zeros_like(poses)
    for i in range(len(aligned_rots)):
        aligned_poses[i] = form_transf(aligned_rots[i], aligned_trans[i])
    return aligned_poses
