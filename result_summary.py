import os
from tqdm import tqdm
import numpy as np
from vo.utils import load_result_poses

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    # data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    save_dir = f"{data_dir}/vo_results/lm"

    if not os.path.exists(f"{data_dir}"):
        print(f"Dataset directory {data_dir} does not exist.")
        exit()

    result_data = np.load(f"{save_dir}/vo_result_poses.npz", allow_pickle=True)
    step = result_data["step"]
    start = result_data["start_idx"]
    last = result_data["last_idx"]
    e_pose, _, gt_all_pose, _, gt_img_pose, _ = load_result_poses(f"{save_dir}/vo_result_poses.npz")

    # Calcularate Absolute Trajectory Error
    error = gt_img_pose - e_pose
    ates = np.sqrt(np.sum(error[:, :3] ** 2, axis=1))
    rmse = np.mean(ates)
    std = np.std(ates)
    print(f"Absolute Trajectory Error\t{rmse:.4f} +/- {std:.4f} [m]")

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

    print(f"Processing time")
    print(f"  Keypoint\t: {kpt_proc_time:.3f} [s]")
    print(f"  Stereo\t: {stereo_proc_time:.3f} [s]")
    print(f"  Optimize\t: {optmize_proc_time:.3f} [s]")



