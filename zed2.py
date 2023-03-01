import cv2
import glob
import numpy as np
from scipy.spatial.transform import Rotation as R
from vo.vo import *
from vo.draw import draw_vo_poses
from vo.detector import *
from vo.utils import *
from vo.datasets.zed2 import *

if __name__ == "__main__":
    data_dir = "./datasets/aki_20230227_2/"
    last_img_idx = len(glob.glob(data_dir + "left/*.png"))
    if last_img_idx == 0:
        raise FileNotFoundError("No images found in the dataset directory.")
    l_imgs, r_imgs = load_images(f"{data_dir}", last_img_idx)
    real_poses, real_quats = read_poses_quats(f"{data_dir}tf_data.csv")
    real_img_poses, real_img_quats = read_camera_pose(f"{data_dir}rover_camera_pose.csv")
    lcam_params, rcam_params = camera_params(f"{data_dir}/camera_params.yaml")

    # 開始画像，終了画像，位置推定間隔を指定
    step = 1
    start = 0
    last = last_img_idx
    # last = 50
    l_imgs = l_imgs[start:last:step]
    r_imgs = r_imgs[start:last:step]
    real_img_poses = real_img_poses[start:last:step]
    real_img_quats = real_img_quats[start:last:step]
    num_img = len(l_imgs)

    # detector = cv2.FastFeatureDetector_create()
    detector = HarrisCornerDetector(blocksize=5, ksize=9, thd=0.01)
    # detector = ShiTomashiCornerDetector()
    # detector = cv2.ORB_create()
    # detector = cv2.AKAZE_create()

    descriptor = cv2.ORB_create()
    # descriptor = cv2.AKAZE_create()
    # descriptor = cv2.SIFT_create()
    # descriptor = cv2.xfeatures2d.SURF_create()

    # Image masking
    D = 50
    img_mask = np.full(l_imgs[0].shape[: 2], 255, dtype=np.uint8)
    img_mask[: D, :] = 0
    img_mask[-100:, :] = 0
    img_mask[:, : D] = 0
    img_mask[:, -D:] = 0

    # Set initial pose
    rot = R.from_quat(real_img_quats[0]).as_matrix()
    trans = np.array([real_img_poses[0]])
    init_pose = np.vstack((np.hstack((rot, trans.T)), np.array([0.0, 0.0, 0.0, 1.0])))

    # vo = MonocularVisualOdometry(lcam_params, l_imgs, detector, descriptor, img_mask=img_mask)
    vo = StereoVisualOdometry(
        lcam_params, rcam_params,
        l_imgs, r_imgs,
        detector, descriptor, img_mask=img_mask, num_disp=50,
        # method="svd",
        method="greedy",
        # use_disp=False
        use_disp=True
    )

    estimated_poses, estimated_quats = vo.estimate_all_poses(init_pose, num_img)

    np.savez(
        f"{data_dir}vo_result_poses.npz",
        estimated_poses=estimated_poses, estimated_quats=estimated_quats,
        real_poses=real_poses, real_quats=real_quats,
        real_img_poses=real_img_poses, real_img_quats=real_img_quats,
        start_idx=start, last_idx=last, step=step
    )
    draw_vo_poses(estimated_poses, real_poses, real_img_poses, view=(-55, 145, -60), ylim=(0.0, 1.0))
    # draw_vo_results(estimated_poses, real_poses, real_img_poses, f"{data_dir}vo_result.png", view=(-55, 145, -60), ylim=(0.0, 1.0))
    vo.save_results(last, start, step, f"{data_dir}/npz/")
