import argparse
import os
from vo.datasets.aki import AkiDataset
from vo.datasets.madmax import MadmaxDataset
from vo.utils import save_trajectory

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert ground truth trajectory to specified format.')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('subdir', help='Subdirectory path')
    parser.add_argument('--format', default='tum', help='Output format', choices=['kitti', 'tum'])
    args = parser.parse_args()

    dataset_dir = f"{DATASET_DIR}/{args.dataset}/{args.subdir}"
    result_dir = f"{dataset_dir}/vo_results/test"

    if args.dataset == "AKI":
        dataset = AkiDataset(dataset_dir)
    else:
        dataset = MadmaxDataset(dataset_dir)

    print(f"Dataset directory: {dataset_dir}")
    gt_ts, gt_poses, gt_quats = dataset.read_all_poses_quats()

    if args.format == 'kitti':
        save_trajectory(f"{dataset_dir}/gt_traj.kitti_f", None, gt_poses, gt_quats, 'kitti')
    elif args.format == 'tum':
        save_trajectory(f"{dataset_dir}/gt_traj.tum_f", gt_ts, gt_poses, gt_quats, 'tum')

    print(f"Saved ground truth trajectory to {dataset_dir}/gt_traj.{args.format}_f")
