import os
import numpy as np
import matplotlib.pyplot as plt
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    # data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    result_dir = f"{data_dir}/vo_results/normal"
    print(f"Result directory: {result_dir}\n")

    est_poses, _, gt_img_poses = load_result_poses(f"{result_dir}/vo_result_poses.npz")
    est_trans = est_poses[:, :3, 3]
    gt_img_trans = gt_img_poses[:, :3, 3]
    trans_errors = est_trans - gt_img_trans

    error_mean = np.mean(trans_errors, axis=0)
    error_std = np.std(trans_errors, axis=0)
    print("Translation errors")
    print(f"\tX : {error_mean[0]} +/- {error_std[0]} [m]")
    print(f"\tY : {error_mean[1]} +/- {error_std[1]} [m]")
    print(f"\tZ : {error_mean[2]} +/- {error_std[2]} [m]")

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(trans_errors[:, 0])
    ax[1].plot(trans_errors[:, 1])
    ax[2].plot(trans_errors[:, 2])
    ax[0].set_ylabel("x [m]")
    ax[1].set_ylabel("y [m]")
    ax[2].set_ylabel("z [m]")
    ax[2].set_xlabel("Frame index")
    plt.show()
    fig.savefig(f"{result_dir}/absolute_trans_error.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

    # draw_trans_diff(est_poses, gt_img_poses, f"{result_dir}/trans_diff.png")
