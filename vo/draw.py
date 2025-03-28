import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import rgb2hex
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.transform import Rotation as R

x_idx = 0
y_idx = 2
z_idx = 1
axis_label = ["x [m]", "y [m]", "z [m]"]


def draw_disparties(disp) -> Figure:
    fig, ax = plt.subplots()
    ax_disp = ax.imshow(disp)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.colorbar(ax_disp)
    return fig


def draw_detected_kpts(img: np.ndarray, kpts: list, descs: list, flag: int = cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS, c: tuple = (0, 255, 0), ps: int = 3) -> np.ndarray:
    kpts = [cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in kpts]
    kpt_img = img.copy()
    if flag == "p" or flag == "points":
        for kpt in kpts:
            cv2.circle(kpt_img, (int(kpt.pt[0]), int(kpt.pt[1])), ps, c, -1)
    else:
        cv2.drawKeypoints(kpt_img, kpts, kpt_img, color=c, flags=flag)
    return kpt_img


def draw_matched_kpts(img: np.ndarray, prev_pts: list[cv2.KeyPoint], curr_pts: list[cv2.KeyPoint]):
    prev_kpts = tuple([cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in prev_pts])
    curr_kpts = tuple([cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in curr_pts])
    match_img = img.copy()
    for prev_kpt, curr_kpt in zip(prev_kpts, curr_kpts):
        cv2.line(match_img, (int(prev_kpt.pt[0]), int(prev_kpt.pt[1])), (int(curr_kpt.pt[0]), int(curr_kpt.pt[1])), (0, 255, 0), 2)
        cv2.circle(match_img, (int(curr_kpt.pt[0]), int(curr_kpt.pt[1])), 1, (0, 0, 255), 3)
        cv2.circle(match_img, (int(prev_kpt.pt[0]), int(prev_kpt.pt[1])), 1, (255, 0, 0), 3)
    return match_img


def draw_matched_kpts_coloring_distance(
        img: np.ndarray,
        prev_pts: list[cv2.KeyPoint], curr_pts: list[cv2.KeyPoint], matches: list[cv2.DMatch],
        src: str = None,
        dm_dist_range: list = None
):
    prev_kpts = tuple([cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in prev_pts])
    curr_kpts = tuple([cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in curr_pts])
    dmatches = tuple([cv2.DMatch(_imgIdx=int(match[0]), _queryIdx=int(match[1]), _trainIdx=int(match[2]), _distance=match[3]) for match in matches])

    dists = [match.distance for match in dmatches]
    cmap = LinearSegmentedColormap.from_list("mycmap", ["blue", "cyan", "yellow", "red"], gamma=1.0)
    if dm_dist_range is not None:
        min_dist, max_dist = dm_dist_range
    else:
        min_dist, max_dist = min(dists), max(dists)
    norm = Normalize(vmin=min_dist, vmax=max_dist)
    match_img = img.copy()
    for match in dmatches:
        x1, y1 = prev_kpts[match.queryIdx].pt
        x2, y2 = curr_kpts[match.trainIdx].pt
        dist = int(match.distance)
        c = rgb2hex(cmap(norm(dist)))
        c = tuple([int(c[5:7], 16), int(c[3:5], 16), int(c[1:3], 16)])
        cv2.line(match_img, (int(x1), int(y1)), (int(x2), int(y2)), c, 1)
        cv2.rectangle(match_img, (int(x1) - 1, int(y1) - 1), (int(x1) + 1, int(y1) + 1), (0, 100, 0), -1)
        cv2.circle(match_img, (int(x2), int(y2)), 2, (255, 0, 255), -1)

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(ScalarMappable(cmap=cmap, norm=norm), cax=cax, shrink=0.8)  # shrinkを入れておくと画像として出力したときの高さが揃う
    if src is not None:
        fig.savefig(src, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def draw_matched_kpts_two_imgs(prev_img: np.ndarray, curr_img: np.ndarray, prev_pts: list[cv2.KeyPoint], curr_pts: list[cv2.KeyPoint], matches: list[cv2.DMatch]):
    prev_kpts = tuple([cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in prev_pts])
    curr_kpts = tuple([cv2.KeyPoint(x=kpt[0], y=kpt[1], size=kpt[2], angle=kpt[3], response=kpt[4], octave=int(kpt[5]), class_id=int(kpt[6])) for kpt in curr_pts])
    dmatches = tuple([cv2.DMatch(_imgIdx=int(match[0]), _queryIdx=int(match[1]), _trainIdx=int(match[2]), _distance=match[3]) for match in matches])
    match_img = cv2.drawMatches(prev_img, prev_kpts, curr_img, curr_kpts, dmatches, None, flags=2)
    return match_img


def draw_vo_poses(
    est_poses: np.ndarray,
    gt_poses: np.ndarray,
    gt_img_poses: np.ndarray = None,
    save_src: str = None,
    draw_data: str = "all",
    dim: int = 3,
    view: tuple[float, float, float] = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    zlim: tuple[float, float] = None,
):
    if dim == 2:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    draw_trajectory(ax, est_poses, gt_img_poses, gt_poses, dim, draw_data)

    if dim == 2:
        ax.set_xlim(xlim) if xlim is not None else None
        ax.set_ylim(zlim) if zlim is not None else None
        ax.set_xlabel(f"{axis_label[x_idx]}")
        ax.set_ylabel(f"{axis_label[y_idx]}")
        ax.set_aspect('equal', adjustable='box')
        ax.grid()
    else:
        set_lims(ax, xlim, ylim, zlim)
        ax.set_xlabel(f"{axis_label[x_idx]}")
        ax.set_ylabel(f"{axis_label[y_idx]}")
        ax.set_zlabel(f"{axis_label[z_idx]}")
        ax.view_init(elev=view[0], azim=view[1], roll=view[2]) if view is not None else None
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    if save_src is not None:
        fig.savefig(save_src, dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.close()
        plt.show()
    else:
        plt.show()


def draw_vo_poses_and_quats(
    est_poses: np.ndarray,
    gt_all_poses: np.ndarray,
    gt_poses: np.ndarray = None,
    scale=0.1,
    save_src: str = None,
    draw_data: str = "all",
    view: tuple[float, float, float] = None,
    xlim: tuple[float, float] = None,
    ylim: tuple[float, float] = None,
    zlim: tuple[float, float] = None,
    step=5
):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    draw_trajectory(ax, est_poses, gt_poses, gt_all_poses, draw_data)

    for e_pose, gt_pose in list(zip(est_poses, gt_poses))[::step]:
        draw_coordinate(ax, e_pose[:3, :3], e_pose[:3, 3], scale=scale)
        draw_coordinate(ax, gt_pose[:3, :3], gt_pose[:3, 3], scale=scale)
    ax.set_xlabel(f"{axis_label[x_idx]}")
    ax.set_ylabel(f"{axis_label[y_idx]}")
    ax.set_zlabel(f"{axis_label[z_idx]}")
    set_lims(ax, xlim, ylim, zlim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.view_init(elev=view[0], azim=view[1], roll=view[2]) if view is not None else None
    fig.tight_layout()
    fig.savefig(save_src, dpi=300, bbox_inches='tight', pad_inches=0) if save_src is not None else None
    plt.show()


def draw_trajectory(
    ax: Axes,
    est_poses: np.ndarray,
    gt_img_poses: np.ndarray,
    gt_poses: np.ndarray,
    dim: int = 3,
    draw_data: str = "all",
    label="Estimated",
):
    est_trans = est_poses[:, :3, 3]
    gt_img_trans = gt_img_poses[:, :3, 3] if gt_img_poses is not None else None
    gt_trans = gt_poses[:, :3, 3] if gt_poses is not None else None
    if draw_data == "all" or draw_data == "truth" or draw_data == "truth_estimated":
        if dim == 2:
            ax.plot(gt_trans[0][x_idx], gt_trans[0][y_idx], 'o', c="r", label="Start")
            ax.plot(gt_trans[-1][x_idx], gt_trans[-1][y_idx], 'x', c="r", label="End")
            ax.plot(gt_trans[:, x_idx], gt_trans[:, y_idx], c='#ff7f0e', label='Truth')
        else:
            ax.plot(gt_trans[0][x_idx], gt_trans[0][y_idx], gt_trans[0][z_idx], 'o', c="r", label="Start")
            ax.plot(gt_trans[-1][x_idx], gt_trans[-1][y_idx], gt_trans[-1][z_idx], 'x', c="r", label="End")
            ax.plot(gt_trans[:, x_idx], gt_trans[:, y_idx], gt_trans[:, z_idx], c='#ff7f0e', label='Truth')
    if draw_data == "all" or draw_data == "estimated" or draw_data == "truth_estimated":
        if dim == 2:
            ax.plot(est_trans[:, x_idx], est_trans[:, y_idx], '-o', label=label, markersize=0, linewidth=2.0)
        else:
            ax.plot(est_trans[:, x_idx], est_trans[:, y_idx], est_trans[:, z_idx], '-o', label=label, markersize=2)
    if draw_data == "all":
        if gt_img_poses is not None:
            if dim == 2:
                ax.plot(gt_img_trans[:, x_idx], gt_img_trans[:, y_idx], 'o', c='#ff7f0e', markersize=2)
            else:
                ax.plot(gt_img_trans[:, x_idx], gt_img_trans[:, y_idx], gt_img_trans[:, z_idx], 'o', c='#ff7f0e', markersize=2)
            for e_pos, r_pos in zip(est_trans, gt_img_trans):
                if dim == 2:
                    ax.plot([e_pos[x_idx], r_pos[x_idx]], [e_pos[y_idx], r_pos[y_idx]], c='r', linewidth=0.3)
                else:
                    ax.plot([e_pos[x_idx], r_pos[x_idx]], [e_pos[y_idx], r_pos[y_idx]], [e_pos[z_idx], r_pos[z_idx]], c='r', linewidth=0.3)


def draw_xyz_pose(ax: list[Axes], poses, label=None):
    if label is not None:
        ax[0].plot(poses[:, 0], label=label)
    else:
        ax[0].plot(poses[:, 0])
    ax[1].plot(poses[:, 1])
    ax[2].plot(poses[:, 2])


def draw_trans_diff(est_poses, gt_poses, save_src=None):
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    est_trans = est_poses[:, :3, 3]
    gt_trans = gt_poses[:, :3, 3]
    draw_xyz_pose(ax, est_trans, label="Estimated")
    draw_xyz_pose(ax, gt_trans, label="Ground truth")
    ax[2].set_xlabel("Frame index")
    ax[0].set_ylabel("X [m]")
    ax[1].set_ylabel("Y [m]")
    ax[2].set_ylabel("Z [m]")
    ax[0].legend()
    plt.show()
    fig.savefig(save_src, dpi=300, bbox_inches='tight', pad_inches=0.1) if save_src else None


def draw_rpy_pose(ax: list[Axes], rpy, label=None):
    if label is not None:
        ax[0].plot(rpy[:, 0], label=label)
    else:
        ax[0].plot(rpy[:, 0])
    ax[1].plot(rpy[:, 1])
    ax[2].plot(rpy[:, 2])


def draw_rpy_diff(e_poses, r_poses, save_src):
    est_rpy = []
    gt_rpy = []
    conv_str = "ZXY"
    for est_pose, gt_pose in zip(e_poses, r_poses):
        est_rot = est_pose[:3, :3]
        gt_rot = gt_pose[:3, :3]

        # TODO Check rover coordinate (Which is X+ direction?)
        est_rpy.append(R.from_matrix(est_rot).as_euler(conv_str, degrees=True))
        gt_rpy.append(R.from_matrix(gt_rot).as_euler(conv_str, degrees=True))
    est_rpy = np.array(est_rpy)
    gt_rpy = np.array(gt_rpy)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    draw_rpy_pose(ax, est_rpy, label="Estimated")
    draw_rpy_pose(ax, gt_rpy, label="Ground truth")
    ax[2].set_xlabel("Frame index")
    ax[0].set_ylabel("Roll [deg]")
    ax[1].set_ylabel("Pitch [deg]")
    ax[2].set_ylabel("Yaw [deg]")
    ax[0].legend()
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
    if trans.shape == (3,):
        trans = trans[:, np.newaxis]
    trans = trans.T[0]
    ax.quiver(trans[x_idx], trans[y_idx], trans[z_idx], xe[x_idx], xe[y_idx], xe[z_idx], color='r')
    ax.quiver(trans[x_idx], trans[y_idx], trans[z_idx], ye[x_idx], ye[y_idx], ye[z_idx], color='g')
    ax.quiver(trans[x_idx], trans[y_idx], trans[z_idx], ze[x_idx], ze[y_idx], ze[z_idx], color='b')


def draw_system_reference_frames(frames: list[np.ndarray], frame_names: list[str] = None, scale=1.0, view=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': '3d'})
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("System reference frames")
    ax.view_init(elev=view[0], azim=view[1], roll=view[2]) if view is not None else None
    ax.set_box_aspect((1, 1, 1))
    set_lims(ax, xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))

    draw_coordinate(ax, np.eye(3), np.array([[0, 0, 0]]).T, scale=scale)
    ax.text(0, 0, 0, "O", size=10, zorder=1, color='k')

    for frame, name in zip(frames, frame_names):
        draw_coordinate(ax, frame[:3, :3], frame[:3, 3:], scale=scale)
        ax.text(frame[0, 3], frame[1, 3], frame[2, 3], name, size=10, zorder=1, color='k')

    plt.show()
