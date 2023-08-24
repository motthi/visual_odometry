import sys
import os
import warnings
import numpy as np
from vo.utils import load_result_poses, poses_to_trans_quats, trans_quats_to_poses, save_trajectory
from vo.draw import draw_vo_poses, draw_vo_poses_and_quats

DATASET_DIR = os.environ['DATASET_DIR']


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


def align(traj, rot, trans):
    aligned_traj = np.zeros_like(traj)
    for i in range(len(traj)):
        aligned_traj[i] = rot @ traj[i] + trans
    return aligned_traj


if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    result_dir = f"{data_dir}/vo_results/normal"
    print(f"Result directory: {result_dir}")

    args = sys.argv
    if len(args) >= 2:
        dim = int(args[1])
        if dim not in [2, 3]:
            warnings.warn("dim is 2 or 3")
            dim = 3
    else:
        dim = 3

    est_ts, est_poses, gt_ts, gt_poses, gt_img_ts, gt_img_poses, = load_result_poses(f"{result_dir}/vo_result_poses.npz")
    est_trans, est_quats = poses_to_trans_quats(est_poses)
    gt_trans, gt_quats = poses_to_trans_quats(gt_poses)
    gt_img_trans, gt_img_quats = poses_to_trans_quats(gt_img_poses)
    R, t, _ = umeyama_alignment(est_trans.T, gt_img_trans.T, with_scale=False, align_start=True)

    est_aligned_trans = align(est_trans, R, t)
    est_aligned_poses = trans_quats_to_poses(est_quats, est_aligned_trans)

    save_trajectory(f"{result_dir}/aligned_est_traj.txt", est_ts, est_aligned_trans, est_quats)
    np.savez(
        f"{result_dir}/aligned_result_poses.npz",
        estimated_timestamps=est_ts, estimated_poses=est_aligned_trans, estimated_quats=est_quats,
        real_timestamps=gt_ts, real_poses=gt_trans, real_quats=gt_quats,
        real_img_timestamps=gt_img_ts, real_img_poses=gt_img_trans, real_img_quats=gt_img_quats
    )

    if dim == 2:
        draw_vo_poses(
            est_poses, gt_poses, gt_img_poses,
            dim=dim,
            draw_data="all",
            # save_src=f"{result_dir}/aligned_trajectory_with_rpy.png",
        )
    else:
        draw_vo_poses_and_quats(
            est_aligned_poses, gt_poses, gt_img_poses,
            draw_data="all",
            scale=0.3,
            step=10,
            # save_src=f"{result_dir}/aligned_trajectory_with_rpy.png",
        )
