import cv2
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from vo.draw import *
from vo.utils import createSaveDirectories
from tqdm import tqdm

if __name__ == "__main__":
    step = 1
    data_dir = "./datasets/aki_20221117_1/"
    # data_dir = "./datasets/feature_less_plane/"

    if not os.path.exists(f"{data_dir}"):
        print(f"Dataset directory {data_dir} does not exist.")
        exit()

    last_img_idx = len(glob.glob(f"{data_dir}left/*.png"))

    createSaveDirectories(data_dir)

    print("Start exporting results...")
    print(f"Dataset directory: {data_dir}")
    for i, last_img_idx in enumerate(tqdm(range(0, last_img_idx - step, step))):
        img = cv2.imread(f"{data_dir}left/{last_img_idx:04d}.png")
        data = np.load(f"{data_dir}npz/{last_img_idx:04d}.npz", allow_pickle=True)

        # Disparities
        if not f"{data['disp']}" == "None":
            fig = draw_disparties(data['disp'])
            fig.savefig(f"{data_dir}disps/{last_img_idx:04d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

        # All detected keypoints
        kpt_img = drawDetectedKeypoints(img, data["kpts"], data["descs"])
        cv2.imwrite(f"{data_dir}kpts/{last_img_idx:04d}.png", kpt_img)

        # Matchd keypoints
        if i == 0:
            continue
        prev_img = cv2.imread(f"{data_dir}left/{last_img_idx-step:04d}.png")
        if prev_img is not None:
            match_img = draw_matched_kpts(prev_img, data["matched_prev_kpts"], data["matched_curr_kpts"])
            cv2.imwrite(f"{data_dir}matched_kpts/{last_img_idx:04d}.png", match_img)
