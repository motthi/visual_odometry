import sys
sys.path.append("../")
import os
from vo.draw import draw_vo_poses_and_quats
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    save_dir = f"{data_dir}/vo_results/normal"
    est_poses, est_quats, gt_poses, gt_quats, gt_img_poses, gt_img_quats = load_result_poses(f"{save_dir}/vo_result_poses.npz")
    draw_vo_poses_and_quats(
        est_poses, est_quats, gt_poses, gt_img_poses, gt_img_quats,
        draw_data="all",
        # view=(-55, 145, -60),
        # xlim=(-2.0, 2.0),
        # ylim=(0.0, 1.0),
        # zlim=(0, 1),
        scale=0.8,
        step=5,
        save_src=f"{save_dir}/quats_result.png",
    )
