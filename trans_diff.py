from vo.draw import draw_trans_diff
from vo.utils import load_result_poses


if __name__ == "__main__":
    data_dir = "./datasets/aki_20221117_1/"
    # data_dir = "./datasets/feature_rich_rock/"
    estimated_poses, _, real_poses, _, real_img_poses, _ = load_result_poses(f"{data_dir}vo_result_poses.npz")
    draw_trans_diff(estimated_poses, real_img_poses, f"{data_dir}trans_diff.png")
