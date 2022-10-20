from vo.utils import draw_vo_results, load_result_poses

if __name__ == "__main__":
    data_dir = "./datasets/aki_20221013_1/"
    # data_dir = "./datasets/feature_rich_rock/"
    estimated_poses, truth_poses, img_poses = load_result_poses(f"{data_dir}vo_result_poses.npz")
    draw_vo_results(
        estimated_poses, truth_poses, img_poses,
        draw_data="truth",
        view=(-55, 145, -60),
        # xlim=(-2.0, 2.0),
        ylim=(0.0, 1.0),
        # zlim=(0, 1),
        save_src=f"{data_dir}poses_truth.png"
        # save_src=f"{data_dir}poses_result.png"
    )
