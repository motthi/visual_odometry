import argparse
import os
from vo.draw import draw_vo_poses, draw_vo_poses_and_quats
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize estimated trajectory.')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--subdir', help='Subdirectory path')
    parser.add_argument('--dim', type=int, help='Dimension of trajectory', default=3)
    parser.add_argument('--rpy', action='store_true', help="Draw RPY")
    parser.add_argument('--aligned', action='store_true', help="Use aligned trajectory")
    args = parser.parse_args()

    data_dir = f"{DATASET_DIR}/{args.dataset}/{args.subdir}"
    result_dir = f"{data_dir}/vo_results/normal"
    print(f"Result directory: {result_dir}")
    print(f"\tDIM\t: {args.dim}")
    print(f"\tRPY\t: {args.rpy}\n")

    if args.aligned:
        npz_src = f"{result_dir}/aligned_result_poses.npz"
    else:
        npz_src = f"{result_dir}/vo_result_poses.npz"
    _, est_poses, _, gt_poses, _, gt_img_poses, = load_result_poses(f"{npz_src}")

    if args.rpy:
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
            dim=args.dim,
            draw_data="all",
            # view=(-55, 145, -60),
            # xlim=(-2.0, 2.0),
            # ylim=(0.0, 1.0),
            # zlim=(0, 1),
            save_src=f"{result_dir}/trajectory_{args.dim}d.png",
        )
