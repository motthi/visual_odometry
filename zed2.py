import cv2
import glob
import quaternion
import numpy as np
from scipy.spatial.transform import Rotation as R
from vo.vo import *
from vo.draw import draw_vo_results
from vo.detector import *
from vo.utils import *
from vo.datasets.zed2 import *

if __name__ == "__main__":
    data_dir = "./datasets/aki_20221117_1/"
    lcam_params, rcam_params = camera_params(f"{data_dir}/camera_params.yaml")
    step = 1
    last_img_idx = len(glob.glob(data_dir + "left/*.png"))
    if last_img_idx == 0:
        raise FileNotFoundError("No images found in the dataset directory.")

    l_imgs, r_imgs = load_images(f"{data_dir}", last_img_idx, step)
    base_rot = np.eye(3)

    # detector = cv2.FastFeatureDetector_create()
    detector = HarrisCornerDetector(blocksize=5, ksize=9, thd=0.001)
    # detector = ShiTomashiCornerDetector()
    # detector = cv2.ORB_create()
    # detector = cv2.AKAZE_create()

    descriptor = cv2.ORB_create()
    # descriptor = cv2.AKAZE_create()
    # descriptor = cv2.SIFT_create()
    # descriptor = cv2.xfeatures2d.SURF_create()

    # vo = MonocularVisualOdometry(lcam_params, l_imgs, detector, descriptor)
    vo = StereoVisualOdometry(
        lcam_params, rcam_params,
        l_imgs, r_imgs,
        detector, descriptor, num_disp=50,
        base_rot=base_rot,
        method="svd",
        use_disp=False
    )

    # Load initial pose
    real_poses, real_quats = read_poses_quats(f"{data_dir}tf_data.csv")
    real_img_poses, real_img_quats = read_camera_pose(f"{data_dir}rover_camera_pose.csv", step)
    rot = R.from_quat(np.array(real_quats[0])).as_matrix()
    trans = np.array([real_poses[0]])
    init_pose = np.vstack((np.hstack((rot, trans.T)), np.array([0.0, 0.0, 0.0, 1.0])))

    estimated_poses = vo.estimate_all_poses(init_pose, last_img_idx, step)

    np.savez(f"{data_dir}vo_result_poses.npz", estimated=estimated_poses, truth=real_poses, img_truth=real_img_poses)
    draw_vo_results(estimated_poses, real_poses, real_img_poses, view=(-55, 145, -60), ylim=(0.0, 1.0))
    # draw_vo_results(estimated_poses, real_poses, real_img_poses, f"{data_dir}vo_result.png", view=(-55, 145, -60), ylim=(0.0, 1.0))
    vo.save_results(last_img_idx, step, f"{data_dir}/results/")
