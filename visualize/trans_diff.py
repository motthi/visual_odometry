import sys
sys.path.append("../")
import os
from vo.draw import draw_trans_diff
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/datasets/aki_20230227_2/"
    estimated_poses, _, real_poses, _, real_img_poses, _ = load_result_poses(f"{data_dir}/vo_result_poses.npz")
    draw_trans_diff(estimated_poses, real_img_poses, f"{data_dir}/trans_diff.png")
