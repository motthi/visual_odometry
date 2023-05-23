import sys
sys.path.append("../")
import os
from vo.draw import draw_euler_diff
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/datasets/aki_20230227_3/"
    _, estimated_quats, _, real_quats, _, real_img_quats = load_result_poses(f"{data_dir}/vo_result_poses.npz")
    draw_euler_diff(estimated_quats, real_img_quats, f"{data_dir}/euler_diff.png")
