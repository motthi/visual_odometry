import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from vo.draw import *
from vo.utils import create_save_directories
from tqdm import tqdm

if __name__ == "__main__":
    data_dir = "./datasets/aki_20230227_2/"

    if not os.path.exists(f"{data_dir}"):
        print(f"Dataset directory {data_dir} does not exist.")
        exit()

    result_data = np.load(f"{data_dir}vo_result_poses.npz", allow_pickle=True)
    step = result_data["step"]
    start = result_data["start_idx"]
    last = result_data["last_idx"]
    create_save_directories(data_dir)

    print("Start exporting results...")
    print(f"Dataset directory: {data_dir}")
    for i, idx in enumerate(tqdm(range(start, last - step, step))):
        img = cv2.imread(f"{data_dir}left/{idx:04d}.png")
        data = np.load(f"{data_dir}npz/{idx:04d}.npz", allow_pickle=True)

        # Disparities
        if 'disp' in data and not f"{data['disp']}" == "None":
            fig = draw_disparties(data['disp'])
            fig.savefig(f"{data_dir}disps/{idx:04d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        # All detected keypoints
        kpt_img = drawDetectedKeypoints(img, data["kpts"], data["descs"])
        cv2.imwrite(f"{data_dir}kpts/{idx:04d}.png", kpt_img)

        # Matchd keypoints
        if i == 0:
            continue
        prev_img = cv2.imread(f"{data_dir}left/{idx-step:04d}.png")
        if prev_img is not None:
            match_img = draw_matched_kpts(prev_img, data["matched_prev_kpts"], data["matched_curr_kpts"])
            cv2.imwrite(f"{data_dir}matched_kpts/{idx:04d}.png", match_img)
