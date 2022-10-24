from __future__ import annotations
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.spatial.transform import Rotation as R


def load_images(src: str = "./datasets", last_img_idx: int = 30, step=1) -> list[np.ndarray, np.ndarray]:
    """Load images from a dataset.

    Args:
        src (str, optional): Dataset directory. Defaults to "./datasets".
        img_num (int, optional): Last image index to be loaded. Defaults to 30.
        step (int, optional): Image index step. Defaults to 1.

    Returns:
        list[np.ndarray, np.ndarray]: Left images and right images.
    """
    l_imgs = []
    r_imgs = []
    for i in range(0, last_img_idx, step):
        l_img = cv2.imread(f"{src}left/{i:04d}.png")
        r_img = cv2.imread(f"{src}right/{i:04d}.png")
        l_imgs.append(l_img)
        r_imgs.append(r_img)
    return l_imgs, r_imgs


def load_result_poses(src: str):
    data = np.load(src)
    estimatde_poses = data['estimated']
    truth_poses = data['truth']
    img_poses = data['img_truth']
    diff = img_poses[0] - estimatde_poses[0]
    estimatde_poses += diff
    return estimatde_poses, truth_poses, img_poses


def draw_vo_results(
    estimated_poses: np.ndarray,
    truth_poses: np.ndarray,
    img_truth_poses: np.ndarray = None,
    save_src: str = None,
    draw_data: str = "all",
    view: tuple[float, float, float] = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    zlim: tuple[float, float] = None,
):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    if draw_data == "all" or draw_data == "truth" or draw_data == "truth_estimated":
        ax.plot(truth_poses[:, 0], truth_poses[:, 1], truth_poses[:, 2], c='#ff7f0e', label='Truth')
        ax.plot(truth_poses[0][0], truth_poses[0][1], truth_poses[0][2], 'o', c="r", label="Start")
        ax.plot(truth_poses[-1][0], truth_poses[-1][1], truth_poses[-1][2], 'x', c="r", label="End")
    if draw_data == "all" or draw_data == "estimated" or draw_data == "truth_estimated":
        ax.plot(estimated_poses[:, 0], estimated_poses[:, 1], estimated_poses[:, 2], '-o', label='Estimated', markersize=2)
    if draw_data == "all":
        if img_truth_poses is not None:
            ax.plot(img_truth_poses[:, 0], img_truth_poses[:, 1], img_truth_poses[:, 2], 'o', c='#ff7f0e', markersize=2)
            for e_pos, r_pos in zip(estimated_poses, img_truth_poses):
                ax.plot([e_pos[0], r_pos[0]], [e_pos[1], r_pos[1]], [e_pos[2], r_pos[2]], c='r', linewidth=0.3)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.view_init(elev=view[0], azim=view[1], roll=view[2]) if view is not None else None
    fig.savefig(save_src, dpi=300, bbox_inches='tight', pad_inches=0) if save_src is not None else None
    plt.show()


def draw_coordinate(ax: Axes, rot: np.ndarray, trans: np.ndarray = np.array([[0, 0, 0]]).T):
    xe = np.array([[1, 0, 0]]).T
    ye = np.array([[0, 1, 0]]).T
    ze = np.array([[0, 0, 1]]).T
    xe = (rot @ xe).T[0]
    ye = (rot @ ye).T[0]
    ze = (rot @ ze).T[0]
    trans = trans.T[0]
    ax.quiver(trans[0], trans[1], trans[2], xe[0], xe[1], xe[2], color='r')
    ax.quiver(trans[0], trans[1], trans[2], ye[0], ye[1], ye[2], color='g')
    ax.quiver(trans[0], trans[1], trans[2], ze[0], ze[1], ze[2], color='b')
