import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from vo.utils import load_result_poses, trajectory_length
from vo.analysis import *

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calcurate RPY error.')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('subdir', help='Subdirectory path')
    parser.add_argument('--aligned', action='store_true', help="Use aligned trajectory")
    args = parser.parse_args()

    data_dir = f"{DATASET_DIR}/{args.dataset}/{args.subdir}"
    result_dir = f"{data_dir}/vo_results/test"
    print(f"Result directory: {result_dir}\n")

    if args.aligned:
        npz_src = f"{result_dir}/aligned_result_poses.npz"
    else:
        npz_src = f"{result_dir}/vo_result_poses.npz"
    _, est_poses, _, _, _, gt_img_poses = load_result_poses(f"{npz_src}")

    # Calcularate Absolute Trajectory Error
    trj_len = trajectory_length(gt_img_poses)
    ate = calc_ate(gt_img_poses, est_poses)
    rpe_trans_list = calc_rpe_trans(gt_img_poses, est_poses)
    rpe_rot_list = calc_rpe_rot(gt_img_poses, est_poses)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(rpe_trans_list)
    axes[1].plot(np.array(rpe_rot_list) * 180.0 / np.pi)
    axes[0].set_ylabel("RTE [m]")
    axes[1].set_ylabel("RRE [deg]")
    axes[1].set_xlabel("Frame index")
    axes[0].grid()
    axes[1].grid()

    # axes[0].set_ylim((-0.00, 0.08))
    # axes[1].set_ylim((-0.00, 4.5))

    fig.tight_layout()
    # fig.savefig(f"{result_dir_base}/rpe.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.show()
