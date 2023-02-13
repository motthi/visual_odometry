from vo.draw import draw_vo_poses
from vo.utils import load_result_poses

if __name__ == "__main__":
    data_dir = "./datasets/aki_20221117_1/"
    # data_dir = "./datasets/feature_rich_rock/"
    estimated_poses, estimated_quats, real_poses, real_quats, real_img_poses, real_img_quats = load_result_poses(f"{data_dir}vo_result_poses.npz")
    draw_vo_poses(
        estimated_poses, real_poses, real_img_poses,
        draw_data="all",
        view=(-55, 145, -60),
        # xlim=(-2.0, 2.0),
        ylim=(0.0, 1.0),
        # zlim=(0, 1),
        # save_src=f"{data_dir}poses_truth.png"
        save_src=f"{data_dir}poses_result.png"
    )
