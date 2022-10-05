import numpy as np
import matplotlib.pyplot as plt


def poses(src: str):
    data = np.load(src)
    estimatde_poses = data['estimated_poses'][1:]
    # img_poses = data['real_poses']
    # diff = img_poses[0] - estimatde_poses[0]
    # estimatde_poses += diff
    return estimatde_poses  # , img_poses


if __name__ == "__main__":
    data_dir = "./datasets/20220929/"
    estimated_poses = poses(f"{data_dir}vo_result_poses.npz")
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(estimated_poses[:, 0], estimated_poses[:, 1], estimated_poses[:, 2], '-o', label='Estimated', markersize=2)
    # ax.plot(img_poses[:, 0], img_poses[:, 1], img_poses[:, 2], '-o', c='#ff7f0e', markersize=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_xlim(4, 8)
    # ax.set_ylim(4, 8)
    ax.set_zlim(-0.2, 0.2)
    plt.show()
