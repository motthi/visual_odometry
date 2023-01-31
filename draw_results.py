import cv2
import copy
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    step = 1
    data_dir = "./datasets/aki_20221025_1/"
    # data_dir = "./datasets/feature_less_plane/"

    if not os.path.exists(f"{data_dir}"):
        print(f"Dataset directory {data_dir} does not exist.")
        exit()

    last_img_idx = len(glob.glob(f"{data_dir}left/*.png"))

    os.makedirs(f"{data_dir}disps/", exist_ok=True)
    os.makedirs(f"{data_dir}kpts/", exist_ok=True)
    os.makedirs(f"{data_dir}matched_kpts/", exist_ok=True)

    print("Start exporting results...")
    print(f"Dataset directory: {data_dir}")
    for i, last_img_idx in enumerate(tqdm(range(0, last_img_idx - step, step))):
        img = cv2.imread(f"{data_dir}left/{last_img_idx:04d}.png")

        data = np.load(f"{data_dir}results/{last_img_idx:04d}.npz", allow_pickle=True)

        # Disparities
        # disp = data['disp']
        # fig, ax = plt.subplots()
        # ax_disp = ax.imshow(disp)
        # ax.axes.xaxis.set_visible(False)
        # ax.axes.yaxis.set_visible(False)
        # plt.colorbar(ax_disp)
        # fig.savefig(f"{data_dir}disps/{last_img_idx:04d}.png", dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.close()

        # All detected keypoints
        kpts = [cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in data["kpts"]]
        descs = data['descs']
        kpt_img = copy.copy(img)
        cv2.drawKeypoints(img, kpts, kpt_img, color=(0, 255, 0), flags=4)
        cv2.imwrite(f"{data_dir}kpts/{last_img_idx:04d}.png", kpt_img)

        if i == 0:
            continue
        prev_img = cv2.imread(f"{data_dir}left/{last_img_idx-step:04d}.png")
        if prev_img is None:
            continue

        # Matchd keypoints
        matches = [cv2.DMatch(_queryIdx=int(m[0]), _trainIdx=int(m[1]), _imgIdx=int(m[2]), _distance=m[3]) for m in data['matches']]
        prev_kpts = tuple([cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in data["matched_prev_kpts"]])
        curr_kpts = tuple([cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in data["matched_curr_kpts"]])
        match_img = cv2.imread(f"{data_dir}left/{last_img_idx:04d}.png")
        for prev_kpt, curr_kpt in zip(prev_kpts, curr_kpts):
            cv2.line(match_img, (int(prev_kpt.pt[0]), int(prev_kpt.pt[1])), (int(curr_kpt.pt[0]), int(curr_kpt.pt[1])), (0, 255, 0), 2)
            cv2.circle(match_img, (int(curr_kpt.pt[0]), int(curr_kpt.pt[1])), 1, (0, 0, 255), 3)
            cv2.circle(match_img, (int(prev_kpt.pt[0]), int(prev_kpt.pt[1])), 1, (255, 0, 0), 3)
        # match_img = cv2.drawMatches(prev_img, prev_kpts, img, curr_kpts, matches, None, flags=2)
        cv2.imwrite(f"{data_dir}matched_kpts/{last_img_idx:04d}.png", match_img)
