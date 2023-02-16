import numpy as np
import matplotlib.pyplot as plt
from vo.utils import load_result_poses
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    data_dir = "./datasets/aki_20221117_1/"
    # data_dir = "./datasets/feature_rich_rock/"
    _, estimated_quats, _, real_quats, _, real_img_quats = load_result_poses(f"{data_dir}vo_result_poses.npz")

    e_eulers = []
    r_eulers = []
    for eq, riq in zip(estimated_quats, real_img_quats):
        # @todo Check rover coordinate (Which is X+ direction?)
        e_eulers.append(R.from_quat(eq).as_euler('ZXY', degrees=True))
        r_eulers.append(R.from_quat(riq).as_euler('ZXY', degrees=True))
    e_eulers = np.array(e_eulers)
    r_eulers = np.array(r_eulers)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(e_eulers[:, 0], label="Estimated")
    ax[0].plot(r_eulers[:, 0], label="Real")
    ax[0].set_ylabel("Roll [deg]")
    ax[0].legend()
    ax[1].plot(e_eulers[:, 1], label="Estimated")
    ax[1].plot(r_eulers[:, 1], label="Real")
    ax[1].set_ylabel("Pitch [deg]")
    ax[1].legend()
    ax[2].plot(e_eulers[:, 2], label="Estimated")
    ax[2].plot(r_eulers[:, 2], label="Real")
    ax[2].set_ylabel("Yaw [deg]")
    ax[2].legend()
    plt.show()
    fig.savefig(f"{data_dir}quats_diff.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
