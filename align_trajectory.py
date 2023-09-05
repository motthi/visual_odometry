import argparse
import os
import numpy as np
from vo.utils import load_result_poses, poses_to_trans_quats, trans_quats_to_poses, save_trajectory, transform_poses, umeyama_alignment
from vo.draw import draw_vo_poses, draw_vo_poses_and_quats

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align estimated trajectory to ground truth trajectory.')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('subdir', help='Subdirectory path')
    parser.add_argument('--dim', help="Dimension")
    args = parser.parse_args()

    if args.dataset is None or args.subdir is None:
        data_dir = f"{DATASET_DIR}/MADMAX/LocationD/D-0"
        save_dir = f"{data_dir}/vo_results/test"
    else:
        data_dir = f"{DATASET_DIR}/{args.dataset}/{args.subdir}"
        save_dir = f"{data_dir}/vo_results/test"
    print(f"Result directory: {save_dir}")

    if args.dim is None:
        dim = 3
    else:
        dim = int(args.dim)

    est_ts, est_poses, gt_ts, gt_poses, gt_img_ts, gt_img_poses, = load_result_poses(f"{save_dir}/vo_result_poses.npz")
    est_trans, est_quats = poses_to_trans_quats(est_poses)
    gt_trans, gt_quats = poses_to_trans_quats(gt_poses)
    gt_img_trans, gt_img_quats = poses_to_trans_quats(gt_img_poses)
    R, t, _ = umeyama_alignment(est_trans.T, gt_img_trans.T, with_scale=False, align_start=True)

    est_aligned_poses = transform_poses(est_poses, R, t)
    est_aligned_trans, est_aligned_quats = poses_to_trans_quats(est_aligned_poses)

    save_trajectory(f"{save_dir}/aligned_est_traj.txt", est_ts, est_aligned_trans, est_aligned_quats)
    np.savez(
        f"{save_dir}/aligned_result_poses.npz",
        est_timestamps=est_ts, est_poses=est_aligned_trans, est_quats=est_aligned_quats,
        gt_timestamps=gt_ts, gt_poses=gt_trans, gt_quats=gt_quats,
        gt_img_timestamps=gt_img_ts, gt_img_poses=gt_img_trans, gt_img_quats=gt_img_quats
    )

    if dim == 2:
        draw_vo_poses(
            est_aligned_poses, gt_poses, gt_img_poses,
            dim=dim,
            draw_data="all",
            # save_src=f"{result_dir}/aligned_trajectory.png",
        )
    else:
        draw_vo_poses_and_quats(
            est_aligned_poses, gt_poses, gt_img_poses,
            draw_data="all",
            scale=0.3,
            step=10,
            # save_src=f"{result_dir}/aligned_trajectory_with_rpy.png",
        )
