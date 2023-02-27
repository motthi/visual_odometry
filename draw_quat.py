from vo.draw import draw_vo_poses_and_quats
from vo.utils import load_result_poses

if __name__ == "__main__":
    data_dir = "./datasets/aki_20230222_1/"
    # data_dir = "./datasets/feature_rich_rock/"
    estimated_poses, estimated_quats, real_poses, real_quats, real_img_poses, real_img_quats = load_result_poses(f"{data_dir}vo_result_poses.npz")
    draw_vo_poses_and_quats(
        estimated_poses, estimated_quats, real_poses, real_img_poses, real_img_quats,
        draw_data="all",
        view=(-55, 145, -60),
        # xlim=(-2.0, 2.0),
        ylim=(0.0, 1.0),
        # zlim=(0, 1),
        save_src=f"{data_dir}quats_result.png"
    )