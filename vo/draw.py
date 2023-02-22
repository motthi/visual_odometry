import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation as R


def draw_disparties(disp) -> Figure:
    fig, ax = plt.subplots()
    ax_disp = ax.imshow(disp)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.colorbar(ax_disp)
    return fig


def drawDetectedKeypoints(img: np.ndarray, kpts: list, descs: list, flag: int = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS, c: tuple = (0, 255, 0), ps: int = 3) -> np.ndarray:
    kpts = [cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in kpts]
    kpt_img = img.copy()
    if flag == "p" or flag == "points":
        for kpt in kpts:
            cv2.circle(kpt_img, (int(kpt.pt[0]), int(kpt.pt[1])), ps, (0, 255, 0), -1)
    else:
        cv2.drawKeypoints(kpt_img, kpts, kpt_img, color=(0, 255, 0), flags=flag)
    return kpt_img


def draw_matched_kpts(img: np.ndarray, prev_pts: list[cv2.KeyPoint], curr_pts: list[cv2.KeyPoint]):
    prev_kpts = tuple([cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in prev_pts])
    curr_kpts = tuple([cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in curr_pts])
    match_img = img.copy()
    for prev_kpt, curr_kpt in zip(prev_kpts, curr_kpts):
        cv2.line(match_img, (int(prev_kpt.pt[0]), int(prev_kpt.pt[1])), (int(curr_kpt.pt[0]), int(curr_kpt.pt[1])), (0, 255, 0), 2)
        cv2.circle(match_img, (int(curr_kpt.pt[0]), int(curr_kpt.pt[1])), 1, (0, 0, 255), 3)
        cv2.circle(match_img, (int(prev_kpt.pt[0]), int(prev_kpt.pt[1])), 1, (255, 0, 0), 3)
    # matches = [cv2.DMatch(_queryIdx=int(m[0]), _trainIdx=int(m[1]), _imgIdx=int(m[2]), _distance=m[3]) for m in data['matches']]
    # match_img = cv2.drawMatches(prev_img, prev_kpts, img, curr_kpts, matches, None, flags=2)
    return match_img


def draw_vo_poses(
    estimated_poses: np.ndarray,
    real_poses: np.ndarray,
    real_img_poses: np.ndarray = None,
    save_src: str = None,
    draw_data: str = "all",
    view: tuple[float, float, float] = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    zlim: tuple[float, float] = None,
):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    draw_trajectory(ax, estimated_poses, real_img_poses, real_poses, draw_data)
    set_lims(ax, xlim, ylim, zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.view_init(elev=view[0], azim=view[1], roll=view[2]) if view is not None else None
    fig.savefig(save_src, dpi=300, bbox_inches='tight', pad_inches=0) if save_src is not None else None
    plt.show()


def draw_vo_poses_and_quats(
    estimated_poses: np.ndarray,
    estimated_quats: np.ndarray,
    real_poses: np.ndarray,
    real_img_poses: np.ndarray = None,
    real_img_quats: np.ndarray = None,
    scale=0.1,
    save_src: str = None,
    draw_data: str = "all",
    view: tuple[float, float, float] = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    zlim: tuple[float, float] = None,
):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    draw_trajectory(ax, estimated_poses, real_img_poses, real_poses, draw_data)
    for e_pose, e_quat, ri_pose, ri_quat in list(zip(estimated_poses, estimated_quats, real_img_poses, real_img_quats))[::5]:
        e_rot = R.from_quat(e_quat).as_matrix()
        ri_rot = R.from_quat(ri_quat).as_matrix()
        draw_coordinate(ax, e_rot, e_pose[:, np.newaxis], scale=scale)
        draw_coordinate(ax, ri_rot, ri_pose[:, np.newaxis], scale=scale)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    # set_lims(ax, xlim, ylim, zlim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.view_init(elev=view[0], azim=view[1], roll=view[2]) if view is not None else None
    fig.savefig(save_src, dpi=300, bbox_inches='tight', pad_inches=0) if save_src is not None else None
    plt.show()


def draw_trajectory(
    ax: Axes,
    estimated_poses: np.ndarray,
    real_img_poses: np.ndarray,
    real_poses: np.ndarray,
    draw_data: str = "all"
):
    if draw_data == "all" or draw_data == "truth" or draw_data == "truth_estimated":
        ax.plot(real_poses[0][0], real_poses[0][1], real_poses[0][2], 'o', c="r", label="Start")
        ax.plot(real_poses[-1][0], real_poses[-1][1], real_poses[-1][2], 'x', c="r", label="End")
        ax.plot(real_poses[:, 0], real_poses[:, 1], real_poses[:, 2], c='#ff7f0e', label='Truth')
    if draw_data == "all" or draw_data == "estimated" or draw_data == "truth_estimated":
        ax.plot(estimated_poses[:, 0], estimated_poses[:, 1], estimated_poses[:, 2], '-o', label='Estimated', markersize=2)
    if draw_data == "all":
        if real_img_poses is not None:
            ax.plot(real_img_poses[:, 0], real_img_poses[:, 1], real_img_poses[:, 2], 'o', c='#ff7f0e', markersize=2)
            for e_pos, r_pos in zip(estimated_poses, real_img_poses):
                ax.plot([e_pos[0], r_pos[0]], [e_pos[1], r_pos[1]], [e_pos[2], r_pos[2]], c='r', linewidth=0.3)


def draw_trans_diff(e_poses, r_poses, save_src=None):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    ax[0].plot(e_poses[:, 0], label="Estimated")
    ax[0].plot(r_poses[:, 0], label="Real")
    ax[0].set_ylabel("X [m]")
    ax[0].legend()
    ax[1].plot(e_poses[:, 1], label="Estimated")
    ax[1].plot(r_poses[:, 1], label="Real")
    ax[1].set_ylabel("Y [m]")
    ax[1].legend()
    ax[2].plot(e_poses[:, 2], label="Estimated")
    ax[2].plot(r_poses[:, 2], label="Real")
    ax[2].set_ylabel("Z [m]")
    ax[2].legend()
    plt.show()
    fig.savefig(save_src, dpi=300, bbox_inches='tight', pad_inches=0.1) if save_src else None


def draw_euler_diff(e_quats, r_quats, save_src):
    e_eulers = []
    r_eulers = []
    conv_str = "ZXY"
    for eq, riq in zip(e_quats, r_quats):
        # @todo Check rover coordinate (Which is X+ direction?)
        e_eulers.append(R.from_quat(eq).as_euler(conv_str, degrees=True))
        r_eulers.append(R.from_quat(riq).as_euler(conv_str, degrees=True))
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
    fig.savefig(save_src, dpi=300, bbox_inches='tight', pad_inches=0.1) if save_src else None


def set_lims(ax: Axes, xlim: tuple[float, float] = None, ylim: tuple[float, float] = None, zlim: tuple[float, float] = None):
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)


def draw_coordinate(ax: Axes, rot: np.ndarray, trans: np.ndarray = np.array([[0, 0, 0]]).T, scale=1.0):
    xe = scale * np.array([[1, 0, 0]]).T
    ye = scale * np.array([[0, 1, 0]]).T
    ze = scale * np.array([[0, 0, 1]]).T
    xe = (rot @ xe).T[0]
    ye = (rot @ ye).T[0]
    ze = (rot @ ze).T[0]
    trans = trans.T[0]
    ax.quiver(trans[0], trans[1], trans[2], xe[0], xe[1], xe[2], color='r')
    ax.quiver(trans[0], trans[1], trans[2], ye[0], ye[1], ye[2], color='g')
    ax.quiver(trans[0], trans[1], trans[2], ze[0], ze[1], ze[2], color='b')
