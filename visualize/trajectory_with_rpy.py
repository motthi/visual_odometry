import sys
sys.path.append("../")
import os
import warnings
from vo.draw import draw_vo_poses_and_quats
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    args = sys.argv
    if len(args) == 2:
        dim = int(args[1])
        if dim not in [2, 3]:
            warnings.warn("dim is 2 or 3")
            dim = 3
    else:
        dim = 3

    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    data_dir = f"{DATASET_DIR}/MADMAX/LocationD/D-0"
    result_dir = f"{data_dir}/vo_results/normal"
    print(f"Result directory: {result_dir}\n")

    est_poses, gt_all_poses, gt_img_poses = load_result_poses(f"{result_dir}/vo_result_poses.npz")
    draw_vo_poses_and_quats(
        est_poses, gt_all_poses, gt_img_poses,
        draw_data="all",
        # view=(-55, 145, -60),
        # xlim=(-2.0, 2.0),
        # ylim=(0.0, 1.0),
        # zlim=(0, 1),
        scale=0.8,
        step=5,
        save_src=f"{result_dir}/trajectory_with_rpy.png",
    )
