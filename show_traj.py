from vo.utils import draw_vo_results, load_result_poses

if __name__ == "__main__":
    data_dir = "./datasets/aki_20221013_1/"
    estimated_poses, real_poses, img_poses = load_result_poses(f"{data_dir}vo_result_poses.npz")
    draw_vo_results(estimated_poses, real_poses, img_poses)
