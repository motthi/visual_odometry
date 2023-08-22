import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    data_dir = f"{DATASET_DIR}/MADMAX/LocationD/D-0"
    result_dir = f"{data_dir}/vo_results/normal"
    print(f"Result directory: {result_dir}\n")

    est_poses, _, gt_img_poses = load_result_poses(f"{result_dir}/vo_result_poses.npz")

    seq = 'ZYX'  # for AKI: 'XYZ', for MADMAX: 'ZYX'
    est_rpys = R.from_matrix(est_poses[:, :3, :3]).as_euler(seq, degrees=True)
    gt_img_rpys = R.from_matrix(gt_img_poses[:, :3, :3]).as_euler(seq, degrees=True)
    errors = est_rpys - gt_img_rpys

    idx = errors > 180
    errors[idx] = errors[idx] - 360
    idx = errors < -180
    errors[idx] = errors[idx] + 360

    error_mean = np.mean(errors, axis=0)
    error_std = np.std(errors, axis=0)
    print("Rotation errors")
    print(f"\tRoll\t: {error_mean[0]} +/- {error_std[0]} [deg]")
    print(f"\tPitch\t: {error_mean[1]} +/- {error_std[1]} [deg]")
    print(f"\tYaw\t: {error_mean[2]} +/- {error_std[2]} [deg]")

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(errors[:, 0])
    ax[1].plot(errors[:, 1])
    ax[2].plot(errors[:, 2])
    ax[0].set_ylabel("Roll [deg]")
    ax[1].set_ylabel("Pitch [deg]")
    ax[2].set_ylabel("Yaw [deg]")
    ax[2].set_xlabel("Frame index")
    plt.show()
    fig.savefig(f"{result_dir}/absolute_rot_error.png", dpi=300, bbox_inches='tight', pad_inches=0.1)

    # draw_rpy_diff(est_poses, gt_img_poses, f"{result_dir}/rpy_diff.png")
