import sys
sys.path.append("../")
import os
from vo.draw import draw_euler_diff
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    # data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    result_dir = f"{data_dir}/vo_results/normal"
    _, est_quats, _, gt_quats, _, gt_img_quats = load_result_poses(f"{result_dir}/vo_result_poses.npz")
    draw_euler_diff(est_quats, gt_img_quats, f"{result_dir}/euler_diff.png")
