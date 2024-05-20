import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calcurate RPY error.')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('subdir', help='Subdirectory path')
    parser.add_argument('--saved_dir', help='Save directory', default="test")
    parser.add_argument('--aligned', action='store_true', help="Use aligned trajectory")
    args = parser.parse_args()

    data_dir = f"{DATASET_DIR}/{args.dataset}/{args.subdir}"
    result_dir = f"{data_dir}/vo_results/{args.saved_dir}"
    print(f"Result directory: {result_dir}\n")

    result_data = np.load(f"{result_dir}/vo_result_poses.npz", allow_pickle=True)
    step = result_data["step"]
    start = result_data["start_idx"]
    last = result_data["last_idx"]

    if args.aligned:
        npz_src = f"{result_dir}/aligned_result_poses.npz"
    else:
        npz_src = f"{result_dir}/vo_result_poses.npz"
    _, est_poses, _, _, _, gt_img_poses = load_result_poses(f"{npz_src}")

    nums_kpts = []
    for i, idx in enumerate(range(start, last - step, step)):
        data = np.load(f"{result_dir}/npz/{idx:05d}.npz", allow_pickle=True)
        nums_kpts.append(len(data["kpts"]))
    nums_kpts = np.array(nums_kpts)

    print(f"Number of keypoints: {nums_kpts.mean():.2f} +- {nums_kpts.std():.2f}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(nums_kpts, label="Number of keypoints")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Number of keypoints")
    ax.legend()
    # fig.savefig(f"{result_dir}/num_kpts.png")
    plt.show()
