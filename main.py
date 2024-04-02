import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from vo.vo import *
from vo.draw import draw_vo_poses
from vo.detector import *
from vo.tracker import *
from vo.utils import poses_to_trans_quats, trans_quats_to_poses, save_trajectory, umeyama_alignment, transform_poses
from vo.method.stereo import *
from vo.datasets.aki import AkiDataset
from vo.datasets.madmax import MadmaxDataset
from vo.datasets.kitti import KittiDataset
from vo.config import *

DATASET_DIR = os.environ['DATASET_DIR']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run visual odometry.')
    parser.add_argument('dataset', help='Dataset name')
    parser.add_argument('subdir', help='Subdirectory path')
    parser.add_argument('--align', action='store_true', help="Align estimated trajectory to ground truth")
    parser.add_argument('--start', type=int, help='Start index of images', default=0)
    parser.add_argument('--last', type=int, help='Last index of images', default=None)
    parser.add_argument('--step', type=int, help='Step of images', default=1)
    parser.add_argument('--save_dir', help='Save directory', default="test")
    args = parser.parse_args()

    if args.dataset == "AKI":
        data_dir = f"{DATASET_DIR}/{args.dataset}/{args.subdir}"
        dataset = AkiDataset(f"{DATASET_DIR}/{args.dataset}/{args.subdir}", start=args.start, last=args.last, step=args.step)
        config_loader = AkiConfigLoader()
    elif args.dataset == "MADMAX":
        data_dir = f"{DATASET_DIR}/{args.dataset}/{args.subdir}"
        dataset = MadmaxDataset(f"{DATASET_DIR}/{args.dataset}/{args.subdir}", start=args.start, last=args.last, step=args.step)
        config_loader = MadmaxConfigLoader()
    elif args.dataset == "KITTI":
        dataset = KittiDataset(f"{DATASET_DIR}/{args.dataset}", args.subdir, start=args.start, last=args.last, step=args.step)
        data_dir = f"{DATASET_DIR}/{args.dataset}/sequences/{int(args.subdir):02d}"

    print(f"Command line arguments")
    print(f"\tDataset\t\t\t: {args.dataset}")
    print(f"\tSubdirectory\t\t: {args.subdir}")
    print(f"\tAlign\t\t\t: {args.align}")
    print(f"\tStart index\t\t: {args.start}")
    print(f"\tLast index\t\t: {args.last}")
    print(f"\tStep\t\t\t: {args.step}")
    print(f"\tSave directory\t\t: {args.save_dir}\n")

    save_dir = f"{data_dir}/vo_results/{args.save_dir}"
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

    config_loader.set(l_imgs[0])

    img_mask = config_loader.data['img_mask']
    detector = config_loader.data['detector']
    descriptor = config_loader.data['descriptor']
    tracker = config_loader.data['tracker']

    # Estimator
    inlier_thd = config_loader.data['inlier_thd']
    max_iter = config_loader.data['max_iter']
    # estimator = MonocularVoEstimator(dataset.lcam_params['intrinsic'])
    # estimator = LmBasedEstimator(dataset.lcam_params['projection'])
    # estimator = RansacLmEstimator(dataset.lcam_params['projection'], max_iter=max_iter, inlier_thd=inlier_thd)
    # estimator = SvdBasedEstimator(dataset.lcam_params['projection'])
    estimator = RansacSvdBasedEstimator(dataset.lcam_params['projection'], max_iter=max_iter, inlier_thd=inlier_thd)
    # estimator = OtsuTwoPointEstimator(dataset.lcam_params, dataset.rcam_params, max_iter=max_iter, inlier_thd=inlier_thd)

    # Stereo matching
    use_disp = config_loader.data['use_disp']
    max_disp = config_loader.data['max_disp']

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
        use_disp=use_disp, max_disp=max_disp
    )

    est_poses = vo.estimate_all_poses(init_pose, num_img)
    est_quats = np.array([R.from_matrix(pose[0:3, 0:3]).as_quat() for pose in est_poses])
    est_trans = np.array([np.array(pose[0:3, 3]).T for pose in est_poses])

    vo.save_results(dataset.last, dataset.start, dataset.step, f"{save_dir}/npz")
    vo.estimator.save_results(f"{save_dir}/estimator_result.npz")
    dataset.save_info(f"{save_dir}/dataset_info.json")
    save_trajectory(f"{save_dir}/estimated_trajectory.txt", img_timestamps, est_trans, est_quats)
    save_trajectory(f"{save_dir}/gt_traj.txt", gt_timestamps, gt_trans, gt_quats, 'tum')
    np.savez(
        f"{save_dir}/vo_result_poses.npz",
        est_timestamps=img_timestamps, est_trans=est_trans, est_quats=est_quats,
        gt_timestamps=gt_timestamps, gt_trans=gt_trans, gt_quats=gt_quats,
        gt_img_timestamps=img_timestamps, gt_img_trans=img_trans, gt_img_quats=img_quats,
        start_idx=dataset.start, last_idx=dataset.last, step=dataset.step
    )

    if args.align:
        rot, t, _ = umeyama_alignment(est_trans.T, img_trans.T, with_scale=False, align_start=True)
        est_aligned_poses = transform_poses(est_poses, rot, t)
        est_aligned_trans, est_aligned_quats = poses_to_trans_quats(est_aligned_poses)

        save_trajectory(f"{save_dir}/aligned_est_traj.txt", img_timestamps, est_aligned_trans, est_quats)
        np.savez(
            f"{save_dir}/aligned_result_poses.npz",
            est_timestamps=img_timestamps, est_trans=est_aligned_trans, est_quats=est_aligned_quats,
            gt_timestamps=gt_timestamps, gt_trans=gt_trans, gt_quats=gt_quats,
            gt_img_timestamps=img_timestamps, gt_img_trans=img_trans, gt_img_quats=img_quats
        )
        est_poses = est_aligned_poses

    draw_vo_poses(
        est_poses, gt_poses, img_poses,
        # view=(-55, 145, -60),
        # ylim=(0.0, 1.0)
    )
