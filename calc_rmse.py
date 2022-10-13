import numpy as np
import matplotlib.pyplot as plt
from vo.utils import *


def relative_pos_diff(estimated_poses, real_poses):
    e_diffs = []
    r_diffs = []
    prev_e_poses = estimated_poses[:-1]
    prev_r_poses = real_poses[1:-1]
    curr_e_poses = estimated_poses[1:]
    curr_r_poses = real_poses[2:]
    for prev_e_pos, prev_r_pos, curr_e_pos, curr_r_pos in zip(prev_e_poses, prev_r_poses, curr_e_poses, curr_r_poses):
        e_diff = np.linalg.norm(curr_e_pos - prev_e_pos)
        r_diff = np.linalg.norm(curr_r_pos - prev_r_pos)
        e_diffs.append(e_diff)
        r_diffs.append(r_diff)
    return e_diffs, r_diffs


if __name__ == "__main__":
    data_dir = "./datasets/aki_20221013_1/"
    estimated_poses, _, img_poses = load_result_poses(f"{data_dir}/vo_result_poses.npz")
    rmses = []
    for e_pos, r_pos in zip(estimated_poses, img_poses):
        rmses.append(np.linalg.norm(e_pos - r_pos))
    e_diffs, r_diffs = relative_pos_diff(estimated_poses, img_poses)

    fig, ax = plt.subplots()
    ax.plot(np.arange(len(e_diffs)), e_diffs, label="Estimated")
    ax.plot(np.arange(len(r_diffs)), r_diffs, label="Truth")
    ax.legend()
    ax.set_xlabel("Image index")
    ax.set_ylabel("Relative position [m]")
    # ax.set_ylim(0, 0.2)
    plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(np.arange(len(rmses)), rmses)
    # ax.set_xlabel("Image Index")
    # ax.set_ylabel("RMSE [m]")
    # plt.show()
    # fig.savefig(f"{data_dir}/rmse.png", dpi=300, bbox_inches='tight', pad_inches=0)
