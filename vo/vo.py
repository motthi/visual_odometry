import cv2
import os
import shutil
import time
import warnings
import numpy as np
from tqdm import tqdm
from vo.utils import form_transf
from vo.datasets import ImageDataset
from vo.method.monocular import VoEstimator
from vo.method.stereo import StereoVoEstimator
from vo.tracker import KeyPointTracker


class VisualOdometry():
    def __init__(
        self,
        dataset: ImageDataset,
        detector, descriptor, tracker: KeyPointTracker, estimator: VoEstimator,
        img_mask: np.ndarray = None
    ) -> None:
        self.ds = dataset
        self.E_l = dataset.lcam_params['extrinsic']
        self.P_l = dataset.lcam_params['projection']
        self.K_l = dataset.lcam_params['intrinsic']
        self.detector = detector
        self.descriptor = descriptor
        self.tracker = tracker
        self.estimator = estimator
        self.img_mask = img_mask
        self.idx = dataset.start
        self.cnt = 0

    def estimate_all_poses(self, init_pose: np.ndarray, last_img_idx: int) -> list:
        print(f"\nStart localization")
        warnings.simplefilter("ignore")

        poses = [init_pose]
        cur_pose = init_pose
        self.cnt = 0

        img_idxes = list(range(self.ds.start, self.ds.last, self.ds.step))
        pbar = tqdm(range(1, last_img_idx))
        for cnt in pbar:
            pt = time.time()
            transf = self.estimate_pose()
            if transf is not None:
                cur_pose = cur_pose @ transf
            poses.append(cur_pose)
            self.cnt += 1
            self.idx = img_idxes[self.cnt]
            ptime = time.time() - pt

            pbar.set_description(f"{cnt}/{last_img_idx-1} ({ptime:.2f}s)")
            if transf is None:
                tqdm.write(f"Index {cnt:03d} : Failed to estimate pose")
        poses = np.array(poses)
        return poses

    def estimate_pose(self):
        raise NotImplementedError

    def detect_track_kpts(self, i: int, prev_img: np.ndarray, curr_img: np.ndarray) -> list[np.ndarray, np.ndarray, np.ndarray]:
        """Detect feature points and track from previous image to current image

        Args:
            i (int): Image index
            curr_img (np.ndarray): current left image

        Returns:
            list[list, list, list]: [Keypoints in previous image, Keypoints in current image, DMatches]
        """
        curr_kpts, curr_descs = self.detect_kpts(curr_img)
        ptp, ctp, dmatches = self.track_kpts(prev_img, curr_img, self.left_kpts[i], self.left_descs[i], curr_kpts, curr_descs)
        return ptp, ctp, dmatches

    def detect_kpts(self, img: np.ndarray) -> list[np.ndarray, np.ndarray]:
        kpts = self.detector.detect(img, self.img_mask)
        kpts, descs = self.descriptor.compute(img, kpts)
        if len(kpts) == 0:
            return [], []
        kpts = np.array(kpts)
        descs = np.array(descs, dtype=np.uint8)
        return kpts, descs

    def track_kpts(
        self,
        prev_img: np.ndarray, curr_img: np.ndarray,
        prev_kpts: np.ndarray, prev_descs: np.ndarray,
        curr_kpts: np.ndarray, curr_descs: np.ndarray
    ) -> list[np.ndarray, np.ndarray, np.ndarray]:
        ptp, ctp, dmatches = self.tracker.track(
            prev_img=prev_img, curr_img=curr_img,
            prev_kpts=prev_kpts, prev_descs=prev_descs,
            curr_kpts=curr_kpts, curr_descs=curr_descs
        )
        return ptp, ctp, dmatches

    def load_img(self, i: int) -> np.ndarray:
        return self.left_imgs[i]


class MonocularVisualOdometry(VisualOdometry):
    def __init__(
        self,
        left_camera_params, left_imgs,
        detector, descriptor, tracker, estimator,
        img_mask=None
    ) -> None:
        super().__init__(left_camera_params, left_imgs, detector, descriptor, tracker, estimator, img_mask)

    def estimate_pose(self):
        # Load images
        left_prev_img = self.load_img(self.idx)
        left_curr_img = self.load_img(self.idx + 1)

        # Detect and track keypoints
        prev_kpts, curr_kpts, dmatches = self.detect_track_kpts(self.idx, left_prev_img, left_curr_img)
        if len(prev_kpts) == 0 or len(curr_kpts) == 0:  # Could not track features
            self.append_kpts_match_info(prev_kpts, curr_kpts, dmatches)
            return None

        transf = self.estimator.estimate(prev_kpts, curr_kpts)
        self.append_kpts_match_info(prev_kpts, curr_kpts, dmatches)
        return transf

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix
        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            T = form_transf(R, t)
            P = np.matmul(np.concatenate((self.K_l, np.zeros((3, 1))), axis=1), T)    # Make the projection matrix
            hom_Q1 = cv2.triangulatePoints(self.P_l, P, q1.T, q2.T)
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1) / np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]  # Make a list of the different possible pairs

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale
        return [R1, t]

    def append_kpts_match_info(self, prev_kpts, curr_kpts, dmatches):
        self.matched_prev_kpts.append(prev_kpts)
        self.matched_curr_kpts.append(curr_kpts)
        self.matches.append(dmatches)

    def save_results(self, last_img_idx: int, start=0, step: int = 1, base_dir: str = "./npz") -> None:
        """Save VO results (Keypoints, Descriptors, Disparity, DMatches, Matched keypoints in previous image, Matched keypoints in current image)

        Args:
            last_img_idx (int): Last image index
            step (int): VO execution step
            base_src (str, optional): Directory to be stored. Defaults to "./npz".
        """
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)
        for i, img_idx in enumerate(range(start, last_img_idx - step, step)):
            kpts = self.left_kpts[i]
            np.savez(
                f"{base_dir}/{img_idx:04d}.npz",
                kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in kpts],
                descs=self.left_descs[i],
                translation=self.Ts[i],
                matches=[[m.queryIdx, m.trainIdx, m.imgIdx, m.distance] for m in self.matches[i]] if self.matches[i] is not None else None,
                matched_prev_kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in self.matched_prev_kpts[i]] if self.matched_prev_kpts[i] is not None else None,
                matched_curr_kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in self.matched_curr_kpts[i]] if self.matched_curr_kpts[i] is not None else None,
            )


class StereoVisualOdometry(VisualOdometry):
    def __init__(
        self,
        dataset: ImageDataset,
        detector, descriptor, tracker: KeyPointTracker, estimator: StereoVoEstimator, img_mask=None,
        num_disp: int = 50,
        method: str = "svd", use_disp: bool = False,
        save_dir: str = "./npz"
    ) -> None:
        super().__init__(dataset, detector, descriptor, tracker, estimator, img_mask)
        self.method = method
        self.use_disp = use_disp
        self.save_dir = save_dir

        # Load camera params
        self.E_r = dataset.rcam_params['extrinsic']
        self.P_r = dataset.rcam_params['projection']
        self.K_r = dataset.rcam_params['intrinsic']

        # Initial processing
        l_img, r_img = self.ds.load_img(0)
        self.prev_kpts, self.prev_descs = self.detect_kpts(l_img)
        if use_disp:
            self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=10)
            self.prev_disp = np.divide(self.disparity.compute(l_img, r_img).astype(np.float32), 16)
        else:
            self.prev_disp = None

    def estimate_pose(self):
        dtc_proc_time = 0.0
        trc_proc_time = 0.0
        stereo_proc_time = 0.0
        opt_proc_time = 0.0

        # Load images
        left_prev_img, _ = self.ds.load_img(self.cnt)
        left_curr_img, right_curr_img = self.ds.load_img(self.cnt + 1)

        # Detect keypoints
        s_time = time.time()
        curr_kpts, curr_descs = self.detect_kpts(left_curr_img)
        prev_kpts, prev_descs = self.prev_kpts, self.prev_descs
        self.prev_kpts, self.prev_descs = curr_kpts, curr_descs
        dtc_proc_time = time.time() - s_time

        # Track keypoints
        s_time = time.time()
        prev_kpts, curr_kpts, dmatches = self.track_kpts(left_prev_img, left_curr_img, prev_kpts, prev_descs, curr_kpts, curr_descs)
        trc_proc_time = time.time() - s_time

        if len(prev_kpts) == 0 or len(curr_kpts) == 0:  # Could not track features
            return None

        # Find the corresponding points in the right image
        s_time = time.time()
        if self.use_disp:   # Calculate disparity
            disp = np.divide(self.disparity.compute(left_curr_img, right_curr_img).astype(np.float32), 16)
        prev_kpts, curr_kpts, dmatches, l_prev_pts, r_prev_pts, l_curr_pts, r_curr_pts = self.find_right_kpts(prev_kpts, curr_kpts, self.prev_disp, disp, dmatches)
        self.prev_disp = disp
        if len(prev_kpts) == 0 or len(curr_kpts) == 0:  # Could not track features
            return None

        # Calculate essential matrix and the correct pose
        prev_3d_pts, curr_3d_pts = self.calc_3d(l_prev_pts, r_prev_pts, l_curr_pts, r_curr_pts)
        stereo_proc_time = time.time() - s_time

        # Estimate pose
        s_time = time.time()
        transf = self.estimator.estimate(l_prev_pts, l_curr_pts, prev_3d_pts, curr_3d_pts)
        opt_proc_time = time.time() - s_time

        self.save_result(
            f"{self.save_dir}/npz/{self.idx:04d}.npz",
            {'detect': dtc_proc_time, 'track': trc_proc_time, 'stereo': stereo_proc_time, 'optimization': opt_proc_time},
            self.prev_kpts, self.prev_descs, self.prev_disp,
            transf, dmatches, prev_kpts, curr_kpts
        )
        return transf

    def find_right_kpts(
            self,
            prev_kpts: list[cv2.KeyPoint], curr_kpts: list[cv2.KeyPoint],
            prev_disps: np.ndarray, curr_disps: np.ndarray, matches: list[cv2.DMatch],
            min_disp: float = 10.0, max_disp: float = 50.0
    ) -> list[list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Find correspond points in the right image and returns keypoints and descriptors in left and right image

        Args:
            prev_kpts (list[cv2.KeyPoint]): List of keypoints in the previous image
            curr_kpts (list[cv2.KeyPoint]): List of keypoints in the current image
            prev_disps (np.ndarray): Keypoint descripoints in the previous image
            curr_disps (np.ndarray): Keypoint descripoints in the current image
            matches (list[cv2.DMatch]): list of DMatches
            min_disp (float, optional): Minimum disparity of stereo processing. Defaults to 10.0.
            max_disp (float, optional): Max disparity of stereo processing. Defaults to 512.0.

        Returns:
            list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]: List of keypoints, descriptors and masks in left and right image
        """

        def get_idxs(q: list, disp):
            q_pts = np.array([q_.pt for q_ in q])
            q_idx = q_pts.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)

        def get_disps(q: list, cnt):
            q_pts = np.array([q_.pt for q_ in q])
            disps = self.stereo_match(q_pts, cnt)
            masks = []
            for disp in disps:
                masks.append(np.logical_and(min_disp < disp, disp < max_disp))
            return np.array(disps), np.array(masks)

        # Get the disparity's for the feature points and mask for min_disp & max_disp
        if self.use_disp:
            prev_disps, p_mask = get_idxs(prev_kpts, prev_disps)
            curr_disps, c_mask = get_idxs(curr_kpts, curr_disps)
        else:
            prev_disps, p_mask = get_disps(prev_kpts, self.idx)
            curr_disps, c_mask = get_disps(curr_kpts, self.idx + 1)
        masks = np.logical_and(p_mask, c_mask)    # Combine the masks

        kpt1_l, kpt2_l, disps1_masked, disps2_masked, matches_masked = [], [], [], [], []
        pq_l, cq_l, pq_r, cq_r = [], [], [], []
        mask_cnt = 0
        for mask, pkpt, ckpt, pdisp, cdisp, match in zip(masks, prev_kpts, curr_kpts, prev_disps, curr_disps, matches):
            if mask:
                pq_l.append(pkpt.pt)
                cq_l.append(ckpt.pt)
                pq_r.append((pkpt.pt[0] - pdisp, pkpt.pt[1]))
                cq_r.append((ckpt.pt[0] - cdisp, ckpt.pt[1]))
                kpt1_l.append(pkpt)
                kpt2_l.append(ckpt)
                disps1_masked.append(pdisp)
                disps2_masked.append(cdisp)
                matches_masked.append(cv2.DMatch(mask_cnt, mask_cnt, match.imgIdx, match.distance))
                mask_cnt += 1
        pq_l = np.array(pq_l)
        cq_l = np.array(cq_l)
        pq_r = np.array(pq_r)
        cq_r = np.array(cq_r)
        return list(kpt1_l), list(kpt2_l), matches_masked, pq_l, pq_r, cq_l, cq_r

    def stereo_match(self, pts, cnt):
        def parabola_subpixel(R, i):
            if R.shape[1] == i + 1:
                return i
            R0 = R[0][i]
            Rp1 = R[0][i + 1]
            Rm1 = R[0][i - 1]
            parabola_d = (Rm1 - Rp1) / (2 * Rm1 - 4 * R0 + 2 * Rp1)
            return i + parabola_d

        def linear_subpixel(R, i):
            if R.shape[1] == i + 1:
                return i
            R0 = R[0][i]
            Rp1 = R[0][i + 1]
            Rm1 = R[0][i - 1]
            if Rp1 < Rm1:
                linear_d = (Rp1 - Rm1) / (2 * (R0 - Rm1))
            else:
                linear_d = (Rp1 - Rm1) / (2 * (R0 - Rp1))
            return i + linear_d

        limg = cv2.cvtColor(self.left_imgs[cnt], cv2.COLOR_BGR2GRAY)
        rimg = cv2.cvtColor(self.right_imgs[cnt], cv2.COLOR_BGR2GRAY)
        WIN_SIZE = 3
        disps = []
        for pt in pts:
            pt_int = (int(pt[0]), int(pt[1]))
            if pt_int[0] < WIN_SIZE or pt_int[0] > limg.shape[1] - WIN_SIZE or pt_int[1] < WIN_SIZE or pt_int[1] > limg.shape[0] - WIN_SIZE:
                disps.append(float('inf'))
                continue
            temp_img = limg[pt_int[1] - WIN_SIZE:pt_int[1] + WIN_SIZE, pt_int[0] - WIN_SIZE:pt_int[0] + WIN_SIZE]
            ref_img = rimg[pt_int[1] - WIN_SIZE:pt_int[1] + WIN_SIZE, :pt_int[0]]
            result = cv2.matchTemplate(ref_img, temp_img, cv2.TM_SQDIFF)
            _, _, loc, _ = cv2.minMaxLoc(result)    # cv2.TM_CCOEFF_NORMEDの場合は第4戻り値を使う
            temp_loc = loc[0]
            temp_loc = parabola_subpixel(result, temp_loc)
            # temp_loc = linear_subpixel(result, temp_loc)
            disp = pt[0] - temp_loc - WIN_SIZE       # テンプレートの中心に来るように補正
            disps.append(disp)
        return disps

    def calc_3d(self, l_fpts_prev: np.ndarray, r_fpts_prev: np.ndarray, l_fpts_curr: np.ndarray, r_fpts_curr: np.ndarray) -> list[np.ndarray]:
        """Calculate 3D position from correspoind points in left and right image

        Args:
            l_fpts_prev (np.ndarray): Feature points in previous left image
            r_fpts_prev (np.ndarray): Feature points in previous right image
            l_fpts_curr (np.ndarray): Feature points in current left image
            r_fpts_curr (np.ndarray): Feature points in current right image

        Returns:
            list: 3D points in previous and current image
        """
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, l_fpts_prev.T, r_fpts_prev.T)  # Triangulate points from i-1'th image
        Q1 = np.transpose(Q1[: 3] / Q1[3])   # Un-homogenize
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, l_fpts_curr.T, r_fpts_curr.T)  # Triangulate points from i'th image
        Q2 = np.transpose(Q2[: 3] / Q2[3])   # Un-homogenize
        return Q1, Q2

    def load_img(self, idx: int) -> list[np.ndarray, np.ndarray]:
        """Load image from the dataset

        Args:
            idx (int): Image index

        Returns:
            list[np.ndarray, np.ndarray]: Left and right image
        """
        l_img = self.left_imgs[idx]
        r_img = self.right_imgs[idx]
        return l_img, r_img

    def save_result(self, src, each_proc_times, kpts, descs, disp, transf, matches=None, matched_prev_kpts=None, matched_curr_kpts=None):
        np.savez(
            src,
            each_process_times=each_proc_times,
            kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in kpts],
            descs=descs,
            disp=disp,
            translation=transf,
            matches=[[m.imgIdx, m.queryIdx, m.trainIdx, m.distance] for m in matches] if matches is not None else None,
            matched_prev_kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in matched_prev_kpts] if matched_prev_kpts is not None else None,
            matched_curr_kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in matched_curr_kpts] if matched_curr_kpts is not None else None
        )
