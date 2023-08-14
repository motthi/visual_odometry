import os
import numpy as np
import matplotlib.pyplot as plt
from vo.utils import *

DATASET_DIR = os.environ['DATASET_DIR']


def relative_pos_diff(estimated_poses, truth_poses):
    e_diffs = []
    r_diffs = []
    prev_e_poses = estimated_poses[:-1]
    prev_r_poses = truth_poses[:-1]
    curr_e_poses = estimated_poses[1:]
    curr_r_poses = truth_poses[1:]
    for prev_e_pos, prev_r_pos, curr_e_pos, curr_r_pos in zip(prev_e_poses, prev_r_poses, curr_e_poses, curr_r_poses):
        e_diff = np.linalg.norm(curr_e_pos - prev_e_pos)
        r_diff = np.linalg.norm(curr_r_pos - prev_r_pos)
        e_diffs.append(e_diff)
        r_diffs.append(r_diff)
    e_diffs = np.array(e_diffs)
    r_diffs = np.array(r_diffs)
    return e_diffs, r_diffs


def draw_relative_position(e_diffs, r_diffs, save_src=None):
    fig, ax = plt.subplots()
    ax.plot(e_diffs, label="Estimated")
    ax.plot(r_diffs, label="Truth")
    ax.set_xlabel("Image index")
    ax.set_ylabel("Relative position [m]")
    ax.legend()
    fig.savefig(save_src, dpi=300, bbox_inches="tight", pad_inches=0.1) if save_src is not None else None
    plt.show()


def draw_relative_position_error(e_diffs, r_diffs, save_src=None):
    fig, ax = plt.subplots()
    relative_error = np.fabs(e_diffs - r_diffs)
    ax.plot(np.arange(relative_error.shape[0]), relative_error)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Relative position error [m]")
    fig.savefig(save_src, dpi=300, bbox_inches="tight", pad_inches=0.1) if save_src is not None else None
    plt.show()


def draw_absolute_position_error(e_pos, r_pos, save_src=None):
    fig, ax = plt.subplots()
    if e_pos.shape[0] != r_pos.shape[0]:
        num = min(e_pos.shape[0], r_pos.shape[0])
        e_pos = e_pos[:num]
        r_pos = r_pos[:num]
    absolute_error = np.linalg.norm(e_pos - r_pos, axis=1)
    ax.plot(np.arange(absolute_error.shape[0]), absolute_error)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Absolute position error [m]")
    fig.savefig(save_src, dpi=300, bbox_inches="tight", pad_inches=0.1) if save_src is not None else None
    plt.show()


def draw_abs_erros(e_poses, r_poses, save_src=None):
    fig, ax = plt.subplots()
    labels = ["Scene B", "Scene C", "Scene D"]
    for e_pose, r_pose, label in zip(e_poses, r_poses, labels):
        absolute_error = np.linalg.norm(e_pose - r_pose, axis=1)
        ax.plot(np.arange(absolute_error.shape[0]), absolute_error, label=label)
        ax.set_xlabel("Image index")
        ax.set_ylabel("Absolute position error [m]")
    ax.legend()
    fig.savefig(save_src, dpi=300, bbox_inches="tight", pad_inches=0.1) if save_src is not None else None
    plt.show()


if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/aki_20221025_4/"
    # data_dir = "./datasets/feature_rich_rock/"
    estimated_poses, _, img_poses = load_result_poses(f"{data_dir}/vo_result_poses.npz")  # FIXME
    rmses = []
    for e_pos, r_pos in zip(estimated_poses, img_poses):
        rmses.append(np.linalg.norm(e_pos - r_pos))
    e_diffs, r_diffs = relative_pos_diff(estimated_poses, img_poses)

    draw_relative_position(e_diffs, r_diffs, save_src=f"{data_dir}/relative_pos.png")
    draw_relative_position_error(e_diffs, r_diffs, save_src=f"{data_dir}/relative_error.png")
    draw_absolute_position_error(estimated_poses, img_poses, save_src=f"{data_dir}/absolute_error.png")
