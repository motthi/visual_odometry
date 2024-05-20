import argparse
import os
import json
import numpy as np
from vo.utils import load_result_poses, trajectory_length
from vo.analysis import calc_ate, calc_roes, calc_rpes

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarize VO results.')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('subdir', help='Subdirectory path')
    parser.add_argument('--saved_dir', help='Save directory', default="test")
    parser.add_argument('--aligned', action='store_true', help="Use aligned trajectory")
    args = parser.parse_args()

    data_dir = f"{DATASET_DIR}/{args.dataset}/{args.subdir}"
    save_dir = f"{data_dir}/vo_results/{args.saved_dir}"
    print(f"Result directory: {save_dir}")

    if not os.path.exists(f"{data_dir}"):
        print(f"Dataset directory {data_dir} does not exist.")
        exit()

    result_data = np.load(f"{save_dir}/vo_result_poses.npz", allow_pickle=True)
    dataset_data = json.load(open(f"{save_dir}/dataset_info.json", "r"))
    step = result_data["step"]
    start = result_data["start_idx"]
    last = result_data["last_idx"]

    if args.aligned:
        _, est_poses, _, gt_poses, _, gt_img_poses = load_result_poses(f"{save_dir}/aligned_result_poses.npz")
    else:
        _, est_poses, _, gt_poses, _, gt_img_poses = load_result_poses(f"{save_dir}/vo_result_poses.npz")

    results = {}

    # Calcularate ATE and RPE
    trj_len = trajectory_length(gt_img_poses)
    ate = calc_ate(gt_img_poses, est_poses)
    rpe_trans = calc_rpes(gt_img_poses, est_poses)
    rpe_rots = calc_roes(gt_img_poses, est_poses)

    accuracy = {}
    rpe_ts = {}
    rpe_rs = {}
    accuracy['ate'] = ate
    for i, (rpe_t, rpe_r) in enumerate(zip(rpe_trans, rpe_rots)):
        rpe_ts[f"{i:05d}"] = rpe_t
        rpe_rs[f"{i:05d}"] = rpe_r
    accuracy['rpe_trans'] = rpe_ts
    accuracy['rpe_rots'] = rpe_rs
    accuracy['rpe_trans_mean'] = np.mean(rpe_trans)
    accuracy['rpe_rots_mean'] = np.mean(rpe_rots)
    accuracy['rpe_trans_std'] = np.std(rpe_trans)
    accuracy['rpe_rots_std'] = np.std(rpe_rots)

    print(f"Trajectory length\t{trj_len:.3f} [m]")
    print("Results")
    print(f"\tATE\t\t{ate} [m]\t({ate/trj_len*100} [%])")
    print(f"\tRPE(trans)\t{np.mean(rpe_trans)}+/-{np.std(rpe_trans)} [m]")
    print(f"\tRPE(rot)\t{np.mean(rpe_rots)*180.0/np.pi}+/-{np.std(rpe_rots)*180.0/np.pi} [deg]")

    # Calculate processing time
    detect_proc_time = 0.0
    track_proc_time = 0.0
    stereo_proc_time = 0.0
    optmize_proc_time = 0.0
    detect_times = {}
    track_times = {}
    stereo_times = {}
    optmize_times = {}
    for i, idx in enumerate(range(start, last - step, step)):
        if i == 0:
            continue
        data = np.load(f"{save_dir}/npz/{idx:05d}.npz", allow_pickle=True)
        proc_times = data['each_process_times'].item()
        detect_proc_time += proc_times['detect']
        track_proc_time += proc_times['track']
        stereo_proc_time += proc_times['stereo']
        optmize_proc_time += proc_times['optimization']
        detect_times[f"{i:05d}"] = proc_times['detect']
        track_times[f"{i:05d}"] = proc_times['track']
        stereo_times[f"{i:05d}"] = proc_times['stereo']
        optmize_times[f"{i:05d}"] = proc_times['optimization']
    detect_proc_time = detect_proc_time / (i + 1)
    track_proc_time = track_proc_time / (i + 1)
    stereo_proc_time = stereo_proc_time / (i + 1)
    optmize_proc_time = optmize_proc_time / (i + 1)

    proc_times['detect'] = detect_times
    proc_times['track'] = track_times
    proc_times['stereo'] = stereo_times
    proc_times['optimization'] = optmize_times
    proc_times['kpt_mean'] = detect_proc_time
    proc_times['track_mean'] = track_proc_time
    proc_times['stereo_mean'] = stereo_proc_time
    proc_times['optimization_mean'] = optmize_proc_time

    print(f"\nProcessing time")
    print(f"  Total\t\t: {detect_proc_time + stereo_proc_time + optmize_proc_time:.5f} [s]")
    print(f"  Keypoint detection\t: {detect_proc_time:.5f} [s]")
    print(f"  Keypoint tracking\t: {track_proc_time:.5f} [s]")
    print(f"  Stereo matching\t: {stereo_proc_time:.5f} [s]")
    print(f"  Optimization\t\t: {optmize_proc_time:.5f} [s]")

    results['trj_len'] = trj_len
    results['accuracy'] = accuracy
    results['process_time'] = proc_times
    json.dump(results, open(f"{save_dir}/summary.json", "w"), indent='\t')

    print("\nFailure information")
    for i, idx in enumerate(range(start, last - step, step)):
        if i == 0:
            continue
        data = np.load(f"{save_dir}/npz/{idx:05d}.npz", allow_pickle=True)
        if data['translation'].shape != (4, 4):
            print(f"Index {i} : Failed")
            print(f"\t{dataset_data['l_img_srcs'][i-1]}")
            print(f"\t{dataset_data['l_img_srcs'][i]}")
