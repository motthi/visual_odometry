import sys
sys.path.append("../")
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from vo.datasets.aki import *
from vo.datasets.madmax import *
from vo.draw import *
from vo.utils import create_save_directories
from tqdm import tqdm

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    # data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    save_dir = f"{data_dir}/vo_results/normal"

    if not os.path.exists(f"{data_dir}"):
        print(f"Dataset directory {data_dir} does not exist.")
        exit()

    result_data = np.load(f"{save_dir}/vo_result_poses.npz", allow_pickle=True)
    step = result_data["step"]
    start = result_data["start_idx"]
    last = result_data["last_idx"]

    dataset = AkiDataset(data_dir, start=start, last=last, step=step)
    # dataset = MadmaxDataset(data_dir, start=start, last=last, step=step)
    
    print("Start exporting results...")
    print(f"Dataset directory: {data_dir}")
    print(f"Save directory: {save_dir}")
    
    create_save_directories(save_dir)
    for i, idx in enumerate(tqdm(range(start, last - step, step))):
        img = cv2.imread(dataset.l_img_srcs[i])
        data = np.load(f"{save_dir}/npz/{idx:04d}.npz", allow_pickle=True)

        # Disparities
        if 'disp' in data and not f"{data['disp']}" == "None":
            fig = draw_disparties(data['disp'])
            fig.savefig(f"{save_dir}/disps/{idx:04d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        # All detected keypoints
        kpt_img = draw_detected_kpts(img, data["kpts"], data["descs"], flag="p")
        cv2.imwrite(f"{save_dir}/kpts/{idx:04d}.png", kpt_img)

        # Matchd keypoints
        if i == 0:
            continue
        prev_img = cv2.imread(dataset.l_img_srcs[i])
        if prev_img is not None:
            match_img = draw_matched_kpts(prev_img, data["matched_prev_kpts"], data["matched_curr_kpts"])
            cv2.imwrite(f"{save_dir}/matched_kpts/{idx:04d}.png", match_img)
