import cv2
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from vo.vo import *
from vo.draw import draw_vo_poses
from vo.detector import *
from vo.tracker import *
from vo.utils import *
from vo.method.stereo import *
from vo.datasets.aki import *
from vo.datasets.madmax import *

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    # Specify the range of images to use
    start = 0
    last = None
    step = 3
    # start = 500
    # last = 2000
    # step = 14

    # Load datasets
    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    dataset = AkiDataset(data_dir, start=start, last=last, step=step)

    # data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    # dataset = MadmaxDataset(data_dir, start=start, last=last, step=step)

    save_dir = f"{data_dir}/vo_results/normal"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    l_imgs, r_imgs = dataset.load_imgs()
    lcam_params, rcam_params = dataset.camera_params()
    _, all_poses, all_quats = dataset.read_all_poses_quats()
    cap_timestamps, cap_poses, cap_quats = dataset.read_captured_poses_quats()
    num_img = len(l_imgs)

    print(f"Dataset directory\t: {data_dir}")
    print(f"Save directory\t\t: {save_dir}")
    print(f"Number of images\t: {num_img}")

    # Feature detector
    # detector = cv2.FastFeatureDetector_create()
    detector = HarrisCornerDetector(blocksize=5, ksize=9, thd=0.005)
    # detector = ShiTomashiCornerDetector()
    # detector = cv2.ORB_create()
    # detector = cv2.AKAZE_create()
    # detector = BucketingDetector(8, 10, detector, cv2.ORB_create())

    # Feature descriptor
    descriptor = cv2.ORB_create()
    # descriptor = cv2.AKAZE_create()
    # descriptor = cv2.SIFT_create()
    # descriptor = cv2.xfeatures2d.SURF_create()

    # Tracker
    tracker = BruteForceTracker(cv2.NORM_HAMMING, cross_check=True)
    # tracker = FlannTracker()
    # tracker = OpticalFlowTracker()

    # Estimator
    # estimator = MonocularVoEstimator(lcam_params['intrinsic'])
    # estimator = LmBasedEstimator(lcam_params['projection'])
    # estimator = SvdBasedEstimator(lcam_params['projection'])
    estimator = RansacSvdBasedEstimator(lcam_params['projection'], max_trial=50, inlier_thd=0.05)

    # Image masking
    img_mask = None
    D = 50
    img_mask = np.full(l_imgs[0].shape[: 2], 255, dtype=np.uint8)
    img_mask[: D, :] = 0
    img_mask[-80:, :] = 0
    img_mask[:, : D] = 0
    img_mask[:, -D:] = 0

    # Set initial pose
    rot = R.from_quat(cap_quats[0]).as_matrix()
    trans = np.array([cap_poses[0]])
    init_pose = np.vstack((np.hstack((rot, trans.T)), np.array([0.0, 0.0, 0.0, 1.0])))

    # vo = MonocularVisualOdometry(lcam_params, l_imgs, detector, descriptor, tracker, estimator, img_mask=img_mask)
    vo = StereoVisualOdometry(
        lcam_params, rcam_params,
        l_imgs, r_imgs,
        detector, descriptor, tracker=tracker,
        estimator=estimator,
        img_mask=img_mask,
        num_disp=100,  # use_disp=True
    )

    estimated_poses, estimated_quats = vo.estimate_all_poses(init_pose, num_img)

    vo.save_results(dataset.last, dataset.start, dataset.step, f"{save_dir}/npz")
    vo.estimator.save_results(f"{save_dir}/estimator_result.npz")
    save_trajectory(f"{save_dir}/estimated_trajectory.txt", cap_timestamps, estimated_poses, estimated_quats)
    np.savez(
        f"{save_dir}/vo_result_poses.npz",
        estimated_poses=estimated_poses, estimated_quats=estimated_quats,
        real_poses=all_poses, real_quats=all_quats,
        real_img_poses=cap_poses, real_img_quats=cap_quats,
        start_idx=dataset.start, last_idx=dataset.last, step=dataset.step
    )
    draw_vo_poses(
        estimated_poses, all_poses, cap_poses,
        view=(-55, 145, -60),
        ylim=(0.0, 1.0)
    )
