import cv2
import numpy as np

FLANN_INDEX_LSH = 6
DMATCH_PT_DIST_THD = 150


class KeyPointTracker(object):
    def track(self) -> list[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


def pickup_good_matches(kpts1, kpts2, matches) -> list[np.ndarray, np.ndarray, np.ndarray]:
    matches = sorted(matches, key=lambda x: x.distance)

    good_pkpts, good_ckpts, good_dmaches = [], [], []

    # for i in range(min(50, len(matches))):
    for cnt in range(len(matches)):
        # TODO: Temporal solution for mismatched keypoints
        pkpt = kpts1[matches[cnt].queryIdx]
        ckpt = kpts2[matches[cnt].trainIdx]
        if np.linalg.norm(np.array(pkpt.pt) - np.array(ckpt.pt)) > DMATCH_PT_DIST_THD:
            continue
        good_pkpts.append(kpts1[matches[cnt].queryIdx])
        good_ckpts.append(kpts2[matches[cnt].trainIdx])
        good_dmaches.append(cv2.DMatch(cnt, cnt, matches[cnt].imgIdx, matches[cnt].distance))
    return np.array(good_pkpts), np.array(good_ckpts), np.array(good_dmaches)


class BruteForceTracker(KeyPointTracker):
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check=True):
        self.bf = cv2.BFMatcher(norm_type, crossCheck=cross_check)

    def track(self, **kwargs) -> list[np.ndarray, np.ndarray, np.ndarray]:
        prev_kpts: np.ndarray = kwargs['prev_kpts']
        prev_descs: np.ndarray = kwargs['prev_descs']
        curr_kpts: np.ndarray = kwargs['curr_kpts']
        curr_descs: np.ndarray = kwargs['curr_descs']

        matches = self.bf.match(prev_descs, curr_descs)

        masked_prev_kpts, masked_curr_kpts, masked_dmatches = pickup_good_matches(prev_kpts, curr_kpts, matches)
        return masked_prev_kpts, masked_curr_kpts, masked_dmatches


class FlannTracker(KeyPointTracker):
    def __init__(
        self,
        index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1),
        search_params=dict(checks=50)
    ):
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def track(self, **kwargs) -> list[np.ndarray, np.ndarray, np.ndarray]:
        prev_kpts: np.ndarray = kwargs['prev_kpts']
        prev_descs: np.ndarray = kwargs['prev_descs']
        curr_kpts: np.ndarray = kwargs['curr_kpts']
        curr_descs: np.ndarray = kwargs['curr_descs']

        matches = self.keypoint_match(prev_descs, curr_descs)

        masked_prev_kpts, masked_curr_kpts, masked_dmatches = pickup_good_matches(prev_kpts, curr_kpts, matches)
        return masked_prev_kpts, masked_curr_kpts, masked_dmatches

    def keypoint_match(self, prev_descs, curr_descs):
        prev_descs.astype(np.float32)
        curr_descs.astype(np.float32)
        matches = self.flann.knnMatch(prev_descs, curr_descs, 2)
        good_matches = []
        for m in matches:
            if len(m) < 2:
                continue
            if m[0].distance < 0.7 * m[1].distance:
                good_matches.append(m[0])
        return good_matches


class OpticalFlowTracker(KeyPointTracker):
    def __init__(
            self,
            win_size=(15, 15),
            flags=cv2.MOTION_AFFINE,
            max_level=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03)):
        self.lk_params = dict(
            winSize=win_size,
            flags=flags,
            maxLevel=max_level,
            criteria=criteria
        )

    def track(self, **kwargs):
        prev_img: np.ndarray = kwargs['prev_img']
        curr_img: np.ndarray = kwargs['curr_img']
        curr_kpts: np.ndarray = kwargs['curr_kpts']

        if len(curr_kpts) == 0:
            return [], [], []

        curr_track_pts = np.expand_dims(cv2.KeyPoint_convert(curr_kpts), axis=1)
        prev_track_pts, st, err = cv2.calcOpticalFlowPyrLK(curr_img, prev_img, curr_track_pts, None, **self.lk_params)
        prev_track_pts = np.around(prev_track_pts)

        trackable = st.astype(bool)
        prev_track_pts = prev_track_pts[trackable]
        curr_track_pts = curr_track_pts[trackable]

        h, w, _ = curr_img.shape
        in_bounds = np.where(np.logical_and(prev_track_pts[:, 1] < h, prev_track_pts[:, 0] < w), True, False)
        prev_track_pts = prev_track_pts[in_bounds]
        curr_track_pts = curr_track_pts[in_bounds]

        prev_kpts = np.array([cv2.KeyPoint(pt[0], pt[1], 1) for pt in prev_track_pts])
        curr_kpts = np.array([cv2.KeyPoint(pt[0], pt[1], 1) for pt in curr_track_pts])

        dmatches = []
        for i in range(len(curr_kpts)):
            dmatches.append(cv2.DMatch(i, i, 0))
        dmatches = np.array(dmatches)

        return prev_kpts, curr_kpts, dmatches
