import numpy as np
import matplotlib.pyplot as plt
from vo.utils import load_result_poses
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    data_dir = "./datasets/aki_20221117_1/"
    # data_dir = "./datasets/feature_rich_rock/"
    estimated_poses, _, real_poses, _, real_img_poses, _ = load_result_poses(f"{data_dir}vo_result_poses.npz")

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(estimated_poses[:, 0], label="Estimated")
    ax[0].plot(real_img_poses[:, 0], label="Real")
    ax[0].set_ylabel("X [m]")
    ax[0].legend()
    ax[1].plot(estimated_poses[:, 1], label="Estimated")
    ax[1].plot(real_img_poses[:, 1], label="Real")
    ax[1].set_ylabel("Y [m]")
    ax[1].legend()
    ax[2].plot(estimated_poses[:, 2], label="Estimated")
    ax[2].plot(real_img_poses[:, 2], label="Real")
    ax[2].set_ylabel("Z [m]")
    ax[2].legend()
    plt.show()
    fig.savefig(f"{data_dir}trans_diff.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
