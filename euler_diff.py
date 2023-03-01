from vo.draw import draw_euler_diff
from vo.utils import load_result_poses


if __name__ == "__main__":
    data_dir = "./datasets/aki_20230227_3/"
    _, estimated_quats, _, real_quats, _, real_img_quats = load_result_poses(f"{data_dir}vo_result_poses.npz")
    draw_euler_diff(estimated_quats, real_img_quats, f"{data_dir}euler_diff.png")
