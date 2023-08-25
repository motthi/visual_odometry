import sys
import os
import warnings
from vo.draw import draw_vo_poses, draw_vo_poses_and_quats
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']


if __name__ == "__main__":
    args = sys.argv
    draw_rpy = False
    if len(args) == 2:
        dim = int(args[1])
        if dim not in [2, 3]:
            warnings.warn("dim is 2 or 3")
            dim = 3
    elif len(args) == 3:
        dim = int(args[1])
        if dim not in [2, 3]:
            warnings.warn("dim is 2 or 3")
            dim = 3
        if args[2] == 'rpy' and dim == 3:
            draw_rpy = True
    else:
        dim = 3
        draw_rpy = False

    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    data_dir = f"{DATASET_DIR}/MADMAX/LocationD/D-0"
    result_dir = f"{data_dir}/vo_results/normal"
    print(f"Result directory: {result_dir}")
    print(f"\tDIM\t: {dim}")
    print(f"\tRPY\t: {draw_rpy}\n")

    _, est_poses, _, gt_poses, _, gt_img_poses, = load_result_poses(f"{result_dir}/vo_result_poses.npz")
    _, est_poses, _, gt_poses, _, gt_img_poses, = load_result_poses(f"{result_dir}/aligned_result_poses.npz")
    if draw_rpy:
        draw_vo_poses_and_quats(
            est_poses, gt_poses, gt_img_poses,
            draw_data="all",
            # view=(-55, 145, -60),
            # xlim=(-2.0, 2.0),
            # ylim=(0.0, 1.0),
            # zlim=(0, 1),
            scale=0.3,
            step=20,
            save_src=f"{result_dir}/trajectory_with_rpy.png",
        )
    else:
        draw_vo_poses(
            est_poses, gt_poses, gt_img_poses,
            dim=dim,
            draw_data="all",
            # view=(-55, 145, -60),
            # xlim=(-2.0, 2.0),
            # ylim=(0.0, 1.0),
            # zlim=(0, 1),
            save_src=f"{result_dir}/trajectory_{dim}d.png",
        )
