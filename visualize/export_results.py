import argparse
import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from vo.datasets.aki import AkiDataset
from vo.datasets.madmax import MadmaxDataset
from vo.datasets.kitti import KittiDataset
from vo.draw import draw_disparties, draw_detected_kpts, draw_matched_kpts, draw_matched_kpts_coloring_distance, draw_matched_kpts_two_imgs
from vo.utils import create_save_directories
from tqdm import tqdm

DATASET_DIR = os.environ['DATASET_DIR']


def dmatch_dist_range(dir, img_idxes):
    dmatch_dists = []
    for idx in img_idxes[1:]:
        data = np.load(f"{dir}/npz/{idx:05d}.npz", allow_pickle=True)
        matches = data["matches"]
        dmatch_dist = [m[3] for m in matches]
        dmatch_dists += dmatch_dist
    return min(dmatch_dists), max(dmatch_dists)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export VO results.')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('subdir', help='Subdirectory path')
    parser.add_argument('--saved_dir', default='test', help='Save directory')
    args = parser.parse_args()

    data_dir = f"{DATASET_DIR}/{args.dataset}/{args.subdir}"
    result_dir = f"{data_dir}/vo_results/{args.saved_dir}"

    # print(args.dataset in ["AKI", "MADMAX"])
    if not os.path.exists(f"{data_dir}"):
        print(f"Dataset directory {data_dir} does not exist.")
        exit()

    result_data = np.load(f"{result_dir}/vo_result_poses.npz", allow_pickle=True)
    step = result_data["step"]
    start = result_data["start_idx"]
    last = result_data["last_idx"]

    dataset_info = json.load(open(f"{data_dir}/vo_results/{args.saved_dir}/dataset_info.json"))
    if "img_idx" in dataset_info:
        img_idxes = dataset_info["img_idx"][:-1]
    else:
        img_idxes = range(start, last - step, step)

    if args.dataset == "AKI":
        dataset = AkiDataset(data_dir, start=start, last=last, step=step, img_idxes=img_idxes)
    elif args.dataset == "MADMAX":
        dataset = MadmaxDataset(data_dir, start=start, last=last, step=step)
    elif args.dataset == "KITTI":
        seq = args.subdir.split("/")[-1]
        dataset = KittiDataset(f"{DATASET_DIR}/{args.dataset}", seq, start=start, last=last, step=step)
    dataset.camera_params()

    dm_dist_range = dmatch_dist_range(result_dir, img_idxes)

    print("Start exporting results...")
    print(f"Dataset directory: {data_dir}")
    print(f"Save directory: {result_dir}")

    create_save_directories(result_dir)
    for i, idx in enumerate(tqdm(img_idxes)):
        img, _ = dataset.load_img(i)
        data = np.load(f"{result_dir}/npz/{idx:05d}.npz", allow_pickle=True)

        # Disparities
        if 'disp' in data and not f"{data['disp']}" == "None":
            fig = draw_disparties(data['disp'])
            fig.savefig(f"{result_dir}/disps/{idx:05d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        # All detected keypoints
        kpt_img = draw_detected_kpts(img, data["kpts"], data["descs"], flag="p", ps=2)
        cv2.imwrite(f"{result_dir}/kpts/{idx:05d}.png", kpt_img)

        # Matchd keypoints
        if i == 0:
            continue
        prev_img, _ = dataset.load_img(i - 1)
        if prev_img is not None:
            draw_matched_kpts_coloring_distance(img, data["matched_prev_kpts"], data["matched_curr_kpts"], data["matches"], f"{result_dir}/matched_kpts/{idx:05d}.png", dm_dist_range)
            # match_img = draw_matched_kpts_two_imgs(prev_img, img, data["matched_prev_kpts"], data["matched_curr_kpts"], data["matches"])
            # match_img = draw_matched_kpts(prev_img, data["matched_prev_kpts"], data["matched_curr_kpts"])
            # cv2.imwrite(f"{result_dir}/matched_kpts/{idx:05d}.png", match_img)
