import numpy as np

# ref https://arxiv.org/pdf/1910.04755.pdf


def calc_ate(gt_poses: np.ndarray, e_poses: np.ndarray) -> float:
    """ Absolute Trajectory Error
    """
    assert len(gt_poses) == len(e_poses)
    errors = []
    for gt_p, e_p in zip(gt_poses, e_poses):
        assert gt_p.shape == e_p.shape == (4, 4)
        E = np.linalg.inv(gt_p) @ e_p
        # print(E)
        # assert np.allclose(E[:3, :3], np.eye(3))
        errors.append(np.linalg.norm(E[:3, 3]))
    return np.mean(errors)


def calc_rpe_trans(gt_poses: np.ndarray, e_poses: np.ndarray) -> float:
    """ Relative Pose Error for translation
    """
    assert len(gt_poses) == len(e_poses)
    num_poses = len(gt_poses)
    errors = []
    for i in range(num_poses - 1):
        Q = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        P = np.linalg.inv(e_poses[i]) @ e_poses[i + 1]
        F = np.linalg.inv(Q) @ P
        # assert np.allclose(F[:3, :3], np.eye(3))
        errors.append(np.linalg.norm(F[:3, 3]))
    return np.mean(errors)


def calc_rpe_rot(gt_poses: np.ndarray, e_poses: np.ndarray) -> float:
    """ Relative Pose Error for rotation
    """
    assert len(gt_poses) == len(e_poses)
    num_poses = len(gt_poses)
    errors = []
    for i in range(num_poses - 1):
        Q = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        P = np.linalg.inv(e_poses[i]) @ e_poses[i + 1]
        F = np.linalg.inv(Q) @ P
        # assert np.allclose(F[:3, :3], np.eye(3))
        ang_r = np.arccos((np.trace(F[:3, :3]) - 1) / 2)
        errors.append(ang_r)
    return np.mean(errors)
