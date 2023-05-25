import cv2
import glob
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from vo.vo import *
from vo.draw import draw_vo_poses
from vo.detector import *
from vo.utils import *
from vo.method.stereo import *
from vo.datasets.aki import *
from vo.datasets.madmax import *

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    # Specify the range of images to use
    start = 540
    last = None
    last = 2000
    step = 14

    # Load datasets
    # data_dir = f"{DATASET_DIR}/AKI/aki_20230227_2"
    # dataset = AkiDataset(data_dir, start=start, last=last, step=step)

    data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    save_dir = f"{data_dir}/vo_result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dataset = MadmaxDataset(data_dir, start=start, last=last, step=step)

    l_imgs, r_imgs = dataset.load_imgs()
    lcam_params, rcam_params = dataset.camera_params()
    all_poses, all_quats = dataset.read_all_poses_quats()
    cap_poses, cap_quats = dataset.read_captured_poses_quats()
    num_img = len(l_imgs)

    # Feature detector
    # detector = cv2.FastFeatureDetector_create()
    # detector = HarrisCornerDetector(blocksize=5, ksize=9, thd=0.08)
    # detector = ShiTomashiCornerDetector()
    detector = cv2.ORB_create()
    # detector = cv2.AKAZE_create()

    # Feature descriptor
    descriptor = cv2.ORB_create()
    # descriptor = cv2.AKAZE_create()
    # descriptor = cv2.SIFT_create()
    # descriptor = cv2.xfeatures2d.SURF_create()

    # Image masking
    img_mask = None
    # D = 50
    # img_mask = np.full(l_imgs[0].shape[: 2], 255, dtype=np.uint8)
    # img_mask[: D, :] = 0
    # img_mask[-100:, :] = 0
    # img_mask[:, : D] = 0
    # img_mask[:, -D:] = 0

    # Set initial pose
    rot = R.from_quat(cap_quats[0]).as_matrix()
    trans = np.array([cap_poses[0]])
    init_pose = np.vstack((np.hstack((rot, trans.T)), np.array([0.0, 0.0, 0.0, 1.0])))

    # vo = MonocularVisualOdometry(lcam_params, l_imgs, detector, descriptor, img_mask=img_mask)
    vo = StereoVisualOdometry(
        lcam_params, rcam_params,
        l_imgs, r_imgs,
        detector, descriptor,
        # estimator=LmBasedEstimator(lcam_params['projection']),
        # estimator=SvdBasedEstimator(lcam_params['projection']),
        estimator=RansacSvdBasedEstimator(lcam_params['projection'], max_trial=50, inlier_thd=0.05),
        matcher=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True),
        img_mask=img_mask,
        use_disp=True,
        num_disp=100,
    )

    estimated_poses, estimated_quats = vo.estimate_all_poses(init_pose, num_img)

    vo.save_results(dataset.last, dataset.start, dataset.step, f"{save_dir}/npz")
    vo.estimator.save_results(f"{save_dir}/estimator_result.npz")
    np.savez(
        f"{save_dir}/vo_result_poses.npz",
        estimated_poses=estimated_poses, estimated_quats=estimated_quats,
        real_poses=all_poses, real_quats=all_quats,
        real_img_poses=cap_poses, real_img_quats=cap_quats,
        start_idx=dataset.start, last_idx=dataset.last, step=dataset.step
    )
    draw_vo_poses(
        estimated_poses, all_poses, cap_poses,
        # view=(-55, 145, -60),
        # ylim=(0.0, 1.0)
    )
