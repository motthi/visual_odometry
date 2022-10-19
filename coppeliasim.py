import cv2
import glob
import quaternion
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from vo.vo import *
from vo.utils import *
from vo.datasets.coppeliasim import *

if __name__ == "__main__":
    data_dir = "./datasets/feature_rich_plane/"
    step = 1
    lcam_params, rcam_params = camera_params()
    last_img_idx = len(glob.glob(data_dir + "left/*.png"))
    l_imgs, r_imgs = load_images(f"{data_dir}", last_img_idx, step)

    vo = StereoVisualOdometry(lcam_params, rcam_params, l_imgs, r_imgs)

    real_poses, real_quats, real_img_poses, real_img_quats = read_poses_quats(f"{data_dir}rover_pose.npz")
    rot = R.from_quat(np.array(real_quats[0])).as_matrix()
    trans = np.array([real_poses[0]])
    init_pose = np.vstack((np.hstack((rot, trans.T)), np.array([0.0, 0.0, 0.0, 1.0])))
    poses = [init_pose]

    for i in tqdm(range(last_img_idx)):
        if i < 1:
            cur_pose = init_pose
        else:
            transf = vo.estimate_pose()
            if transf is None:
                print("Cannot estimate pose")
                continue
            cur_pose = np.matmul(cur_pose, transf)
            poses.append(cur_pose)

    estimated_poses = np.array([np.array(pose[0:3, 3]).T for pose in poses])
    np.savez(f"{data_dir}vo_result_poses.npz", estimated_poses=estimated_poses, real_poses=real_poses, real_img_poses=real_img_poses)
    vo.save_results(last_img_idx, step, f"{data_dir}/results/")
    draw_vo_results(estimated_poses, real_poses, real_img_poses, save_src=f"{data_dir}results.png")
    # draw_vo_results(estimated_poses, real_poses, real_img_poses)
