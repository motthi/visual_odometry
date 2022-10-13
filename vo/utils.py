import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def draw_vo_results(
    estimated_poses: np.ndarray,
    real_poses: np.ndarray,
    img_real_poses: np.ndarray = None,
    save_src: str = None
):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(real_poses[:, 0], real_poses[:, 1], real_poses[:, 2], c='#ff7f0e', label='Truth')
    ax.plot(real_poses[0][0], real_poses[0][1], real_poses[0][2], 'o', c="r", label="Start")
    ax.plot(real_poses[-1][0], real_poses[-1][1], real_poses[-1][2], 'x', c="r", label="End")
    ax.plot(img_real_poses[:, 0], img_real_poses[:, 1], img_real_poses[:, 2], 'o', c='#ff7f0e', markersize=2) if img_real_poses is not None else None
    ax.plot(estimated_poses[:, 0], estimated_poses[:, 1], estimated_poses[:, 2], '-o', label='Estimated', markersize=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_xlim(-2, 2)
    # ax.set_ylim(-2, 2)
    # ax.set_zlim(-2, 2)
    fig.savefig(save_src, dpi=300, bbox_inches='tight', pad_inches=0) if save_src is not None else None
    plt.show()
