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
    # start = 2000
    # last = 2500
    # step = 7

    # Load datasets
    data_dir = f"{DATASET_DIR}/AKI/aki_20230615_1"
    dataset = AkiDataset(data_dir, start=start, last=last, step=step)

    # data_dir = f"{DATASET_DIR}/MADMAX/LocationA/A-0"
    # dataset = MadmaxDataset(data_dir, start=start, last=last, step=step)

    save_dir = f"{data_dir}/vo_results/normal"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gt_timestamps, gt_trans, gt_quats = dataset.read_all_poses_quats()
    img_timestamps, img_trans, img_quats = dataset.read_captured_poses_quats()
    gt_poses = trans_quats_to_poses(gt_quats, gt_trans)
    img_poses = trans_quats_to_poses(img_quats, img_trans)
    l_imgs, r_imgs = dataset.load_imgs()
    num_img = len(l_imgs)

    print(f"Dataset directory\t: {data_dir}")
    print(f"Save directory\t\t: {save_dir}")
    print(f"Number of images\t: {num_img}")

    # Feature detector
    # detector = cv2.FastFeatureDetector_create()
    # detector = HarrisCornerDetector(blocksize=5, ksize=5, thd=0.01)
    # detector = ShiTomashiCornerDetector()
    detector = cv2.ORB_create()
    # detector = cv2.AKAZE_create()

    # Feature descriptor
    descriptor = cv2.ORB_create()
    # descriptor = cv2.AKAZE_create()
    # descriptor = cv2.SIFT_create()
    # descriptor = cv2.xfeatures2d.SURF_create()

    # Tracker
    tracker = BruteForceTracker(cv2.NORM_HAMMING, cross_check=True)
    # tracker = FlannTracker()
    # tracker = OpticalFlowTracker(win_size=(100, 100))

    # Estimator
    # estimator = MonocularVoEstimator(dataset.lcam_params['intrinsic'])
    estimator = LmBasedEstimator(dataset.lcam_params['projection'])
    # estimator = SvdBasedEstimator(dataset.lcam_params['projection'])
    # estimator = RansacSvdBasedEstimator(dataset.lcam_params['projection'], max_trial=50, inlier_thd=0.05)

    # Image masking
    img_mask = None
    img_mask = np.full(l_imgs[0].shape[: 2], 255, dtype=np.uint8)
    # img_mask[300:, :] = 0
    # D = 50
    # img_mask = np.full(l_imgs[0].shape[: 2], 255, dtype=np.uint8)
    # img_mask[: D, :] = 0
    # img_mask[-80:, :] = 0
    # img_mask[:, : D] = 0
    # img_mask[:, -D:] = 0

    # Set initial pose
    rot = R.from_quat(img_quats[0]).as_matrix()
    trans = np.array([img_trans[0]])
    init_pose = form_transf(rot, trans)

    # vo = MonocularVisualOdometry(dataset.lcam_params, l_imgs, detector, descriptor, tracker, estimator, img_mask=img_mask)
    vo = StereoVisualOdometry(
        dataset.lcam_params, dataset.rcam_params,
        l_imgs, r_imgs,
        detector, descriptor, tracker=tracker,
        estimator=estimator,
        img_mask=img_mask,
        # use_disp=True
    )

    est_poses = vo.estimate_all_poses(init_pose, num_img)
    est_quats = np.array([R.from_matrix(pose[0:3, 0:3]).as_quat() for pose in est_poses])
    est_trans = np.array([np.array(pose[0:3, 3]).T for pose in est_poses])

    vo.save_results(dataset.last, dataset.start, dataset.step, f"{save_dir}/npz")
    vo.estimator.save_results(f"{save_dir}/estimator_result.npz")
    save_trajectory(f"{save_dir}/estimated_trajectory.txt", img_timestamps, est_trans, est_quats)
    save_trajectory(f"{save_dir}/gt_traj.txt", gt_timestamps, gt_trans, gt_quats, 'tum')
    np.savez(
        f"{save_dir}/vo_result_poses.npz",
        estimated_poses=est_trans, estimated_quats=est_quats,
        real_poses=gt_trans, real_quats=gt_quats,
        real_img_poses=img_trans, real_img_quats=img_quats,
        start_idx=dataset.start, last_idx=dataset.last, step=dataset.step
    )
    draw_vo_poses(
        est_poses, gt_poses, img_poses,
        view=(-55, 145, -60),
        ylim=(0.0, 1.0)
    )
