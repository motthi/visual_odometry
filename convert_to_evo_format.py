import os
from vo.datasets.aki import *
from vo.datasets.madmax import *
from vo.utils import save_trajectory

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    dataset_dir = f"{DATASET_DIR}/AKI/aki_20230227_2"
    dataset = AkiDataset(dataset_dir)

    dataset_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    dataset = MadmaxDataset(dataset_dir)

    print(f"Dataset directory: {dataset_dir}")
    gt_ts, gt_poses, gt_quats = dataset.read_all_poses_quats()

    # save_trajectory(f"{dataset_dir}/gt_traj.kitti_f", None, gt_poses, gt_quats, 'kitti')
    save_trajectory(f"{dataset_dir}/gt_traj.tum_f", gt_ts, gt_poses, gt_quats, 'tum')
