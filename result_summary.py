import os
import json
import numpy as np
from vo.utils import load_result_poses, trajectory_length
from vo.analysis import calc_ate, calc_rpe_rot, calc_rpe_trans

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    # data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    save_dir = f"{data_dir}/vo_results/normal"

    if not os.path.exists(f"{data_dir}"):
        print(f"Dataset directory {data_dir} does not exist.")
        exit()

    result_data = np.load(f"{save_dir}/vo_result_poses.npz", allow_pickle=True)
    dataset_data = json.load(open(f"{save_dir}/dataset_info.json", "r"))
    step = result_data["step"]
    start = result_data["start_idx"]
    last = result_data["last_idx"]

    _, est_poses, _, gt_poses, _, gt_img_poses = load_result_poses(f"{save_dir}/vo_result_poses.npz")
    _, est_poses, _, gt_poses, _, gt_img_poses = load_result_poses(f"{save_dir}/aligned_result_poses.npz")

    # Calcularate ATE and RPE
    trj_len = trajectory_length(gt_img_poses)
    ate = calc_ate(gt_img_poses, est_poses)
    rpe_trans = calc_rpe_trans(gt_img_poses, est_poses)
    rpe_rot = calc_rpe_rot(gt_img_poses, est_poses)
    print(f"Trajectory length\t{trj_len:.3f} [m]")
    print("Results")
    print(f"\tATE\t\t{ate} [m]\t({ate/trj_len*100} [%])")
    print(f"\tRPE(trans)\t{rpe_trans} [m]")
    print(f"\tRPE(rot)\t{rpe_rot*180.0/np.pi} [deg]")

    # Calculate processing time
    kpt_proc_time = 0.0
    stereo_proc_time = 0.0
    optmize_proc_time = 0.0
    for i, idx in enumerate(range(start, last - step, step)):
        if i == 0:
            continue
        data = np.load(f"{save_dir}/npz/{idx:04d}.npz", allow_pickle=True)
        proc_times = data['each_process_times'].item()
        kpt_proc_time += proc_times['kpt']
        stereo_proc_time += proc_times['stereo']
        optmize_proc_time += proc_times['optimization']
    kpt_proc_time = kpt_proc_time / (i + 1)
    stereo_proc_time = stereo_proc_time / (i + 1)
    optmize_proc_time = optmize_proc_time / (i + 1)

    print(f"\nProcessing time")
    print(f"  Total\t\t: {kpt_proc_time + stereo_proc_time + optmize_proc_time:.5f} [s]")
    print(f"  Keypoint\t: {kpt_proc_time:.5f} [s]")
    print(f"  Stereo\t: {stereo_proc_time:.5f} [s]")
    print(f"  Optimize\t: {optmize_proc_time:.5f} [s]")

    print("\nFailure information")
    for i, idx in enumerate(range(start, last - step, step)):
        if i == 0:
            continue
        data = np.load(f"{save_dir}/npz/{idx:04d}.npz", allow_pickle=True)
        if data['translation'].shape != (4, 4):
            print(f"Index {i} : Failed")
            print(f"\t{dataset_data['l_img_srcs'][i-1]}")
            print(f"\t{dataset_data['l_img_srcs'][i]}")
