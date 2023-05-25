import sys
sys.path.append("../")
import os
from vo.draw import draw_euler_diff
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/datasets/aki_20230227_3/"
    data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    save_dir = f"{data_dir}/vo_result"
    _, estimated_quats, _, real_quats, _, real_img_quats = load_result_poses(f"{save_dir}/vo_result_poses.npz")
    draw_euler_diff(estimated_quats, real_img_quats, f"{save_dir}/euler_diff.png")
