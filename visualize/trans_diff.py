import sys
sys.path.append("../")
import os
from vo.draw import draw_trans_diff
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    # data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    result_dir = f"{data_dir}/vo_results/normal"
    est_poses, _, gt_poses, _, gt_img_poses, _ = load_result_poses(f"{result_dir}/vo_result_poses.npz")  # FIXME
    draw_trans_diff(est_poses, gt_img_poses, f"{result_dir}/trans_diff.png")
