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
        # assert np.allclose(E[:3, :3], np.eye(3))
        # errors.append(np.linalg.norm(E[:1, 3]))
        errors.append(np.linalg.norm(E[:3, 3]))
    return np.mean(errors)


def calc_rpes(gt_poses: np.ndarray, e_poses: np.ndarray) -> np.ndarray:
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
        errors.append(calc_tran(F))
    return np.array(errors)


def calc_roes(gt_poses: np.ndarray, e_poses: np.ndarray) -> np.ndarray:
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
        errors.append(calc_angle(F))
    return np.array(errors)


def calc_tran(M: np.ndarray):
    assert M.shape == (4, 4)
    return np.linalg.norm(M[:3, 3])


def calc_angle(M: np.ndarray):
    assert M.shape == (4, 4)
    return np.arccos((np.trace(M[:3, :3]) - 1) / 2)
