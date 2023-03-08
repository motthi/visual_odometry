from __future__ import annotations
import cv2
import os
import shutil
import time
import warnings
import numpy as np
from tqdm import tqdm
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from vo.matcher import FlannMatcher

# https://github.com/niconielsen32/ComputerVision/tree/a3caf60f0134704958879b9c7e3ef74090ca6579/VisualOdometry

MAX_MATCH_PTS = 50


class VisualOdometry():
    def __init__(
            self,
            camera_params, imgs,
            detector, descriptor, matcher,
        img_mask
    ) -> None:
        self.E_l = camera_params['extrinsic']
        self.P_l = camera_params['projection']
        self.K_l = camera_params['intrinsic']
        self.detector = detector
        self.descriptor = descriptor
        self.matcher = matcher
        self.left_imgs = imgs
        self.img_mask = img_mask
        self.cnt = 0

        l_img = imgs[0]
        l_kpts = self.detector.detect(l_img, self.img_mask)
        l_kpts, l_descs = self.descriptor.compute(l_img, l_kpts)
        l_descs = np.array(l_descs, dtype=np.uint8)

        self.Ts = [None]
        self.left_kpts = [l_kpts]
        self.left_descs = [l_descs]
        self.matches = [None]
        self.matched_prev_kpts = [None]
        self.matched_curr_kpts = [None]
        self.process_times = [None]

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = -t
        T[3, 3] = 1.0
        return T

    def estimate_all_poses(self, init_pose: np.ndarray, last_img_idx: int) -> list:
        warnings.simplefilter("ignore")
        poses = [init_pose]
        self.process_times = [None]
        cur_pose = init_pose
        for idx in tqdm(range(1, last_img_idx)):
            s_time = time.time()
            transf = self.estimate_pose()
            self.Ts.append(transf)
            if transf is not None:
                cur_pose = cur_pose @ transf
            self.process_times.append(time.time() - s_time)
            poses.append(cur_pose)
            self.cnt += 1
            if transf is None:
                tqdm.write(f"Index {idx:03d} : Failed to estimate pose")
        quats = np.array([R.from_matrix(pose[0:3, 0:3]).as_quat() for pose in poses])
        poses = np.array([np.array(pose[0:3, 3]).T for pose in poses])
        return poses, quats

    def estimate_pose(self):
        raise NotImplementedError

    def detect_track_kpts(self, i: int, curr_img: np.ndarray) -> list[np.ndarray, np.ndarray, np.ndarray]:
        """Detect feature points and track from previous image to current image

        Args:
            i (int): Image index
            curr_img (np.ndarray): current left image

        Returns:
            list[list, list, list]: [Keypoints in previous image, Keypoints in current image, DMatches]
        """
        curr_kpts, curr_descs = self.detect_kpts(curr_img)
        tp1, tp2, dmatches = self.track_kpts(i, curr_kpts, curr_descs)
        return tp1, tp2, dmatches

    def detect_kpts(self, img: np.ndarray) -> list[np.ndarray, np.ndarray]:
        kpts = self.detector.detect(img, self.img_mask)
        kpts, descs = self.descriptor.compute(img, kpts)
        if len(kpts) == 0:
            self.left_kpts.append(kpts)
            self.left_descs.append(descs)
            return [], []
        descs = np.array(descs, dtype=np.uint8)
        self.left_kpts.append(kpts)
        self.left_descs.append(descs)
        return kpts, descs

    def track_kpts(self, i: int, curr_kpts: np.ndarray, curr_descs: np.ndarray) -> list[np.ndarray, np.ndarray, np.ndarray]:
        prev_kpts = self.left_kpts[i]
        prev_descs = self.left_descs[i]
        matches = self.matcher.match(prev_descs, curr_descs)

        masked_prev_kpts, masked_curr_kpts, masked_dmatches = [], [], []
        matches = sorted(matches, key=lambda x: x.distance)
        # for i in range(min(50, len(matches))):
        for i in range(len(matches)):
            masked_prev_kpts.append(prev_kpts[matches[i].queryIdx])
            masked_curr_kpts.append(curr_kpts[matches[i].trainIdx])
            masked_dmatches.append(cv2.DMatch(i, i, matches[i].imgIdx, matches[i].distance))
        return np.array(masked_prev_kpts), np.array(masked_curr_kpts), np.array(masked_dmatches)

    def load_img(self, i: int) -> np.ndarray:
        return self.left_imgs[i]


class MonocularVisualOdometry(VisualOdometry):
    def __init__(
            self,
            left_camera_params, left_imgs,
            detector, descriptor, matcher=FlannMatcher(),
            img_mask=None
    ) -> None:
        super().__init__(left_camera_params, left_imgs, detector, descriptor, matcher, img_mask)

    def estimate_pose(self):
        # Load images
        left_curr_img = self.load_img(self.cnt + 1)

        # Detect and track keypoints
        prev_kpts, curr_kpts, dmatches = self.detect_track_kpts(self.cnt, left_curr_img)

        if len(prev_kpts) == 0 or len(curr_kpts) == 0:  # Could not track features
            self.append_kpts_match_info(prev_kpts, curr_kpts, dmatches)
            return None

        transf = self.estimate_transform_matrix(prev_kpts, curr_kpts)
        self.append_kpts_match_info(prev_kpts, curr_kpts, dmatches)
        return transf

    def estimate_transform_matrix(self, pkpts: list(cv2.KeyPoint), ckpts: list(cv2.KeyPoint)):
        """
        Calculates the transformation matrix
        Parameters
        ----------
        pkpts (ndarray): The good keypoints matches position in i-1'th image
        ckpts (ndarray): The good keypoints matches position in i'th image
        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        pkpts = np.float32([kpt.pt for kpt in pkpts])
        ckpts = np.float32([kpt.pt for kpt in ckpts])
        # pkpts = cv2.undistortPoints(np.expand_dims(pkpts, axis=1), cameraMatrix=self.K_l, distCoeffs=None)
        # ckpts = cv2.undistortPoints(np.expand_dims(ckpts, axis=1), cameraMatrix=self.K_l, distCoeffs=None)

        E, mask = cv2.findEssentialMat(pkpts, ckpts, cameraMatrix=self.K_l, threshold=1)
        if E.shape != (3, 3):
            return None
        _, R, t, _ = cv2.recoverPose(E, pkpts, ckpts, cameraMatrix=self.K_l, mask=mask)

        # scale = 0.18
        # t = scale * t

        # R, t = self.decomp_essential_mat(E, pkpts, ckpts)
        transf = self._form_transf(R, np.squeeze(t))
        # transf = np.linalg.inv(transf)
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
            T = self._form_transf(R, t)
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
                process_time=self.process_times[i],
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
        left_camera_params, right_camera_params, left_imgs, right_imgs,
        detector, descriptor, matcher=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True), img_mask=None,
        num_disp: int = 300,
        method: str = "svd", use_disp: bool = False
    ) -> None:
        super().__init__(left_camera_params, left_imgs, detector, descriptor, matcher, img_mask)
        self.right_imgs = right_imgs
        self.method = method
        self.use_disp = use_disp

        # Load camera params
        self.E_r = right_camera_params['extrinsic']
        self.P_r = right_camera_params['projection']
        self.K_r = right_camera_params['intrinsic']

        # Initial processing
        l_img, r_img = self.load_img(self.cnt)
        if use_disp:
            self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=10)
            self.disparities = [np.divide(self.disparity.compute(l_img, r_img).astype(np.float32), 16)]
        else:
            self.disparities = [None]

    def estimate_pose(self):
        # Load images
        left_curr_img, right_curr_img = self.load_img(self.cnt + 1)

        # Calculate disparity
        if self.use_disp:
            self.disparities.append(np.divide(self.disparity.compute(left_curr_img, right_curr_img).astype(np.float32), 16))
        else:
            self.disparities.append(None)

        # Detect and track keypoints
        prev_kpts, curr_kpts, dmatches = self.detect_track_kpts(self.cnt, left_curr_img)
        if len(prev_kpts) == 0 or len(curr_kpts) == 0:  # Could not track features
            self.append_kpts_match_info(prev_kpts, curr_kpts, dmatches)
            return None

        # Find the corresponding points in the right image
        prev_kpts, curr_kpts, dmatches, l_prev_pts, r_prev_pts, l_curr_pts, r_curr_pts = self.find_right_kpts(prev_kpts, curr_kpts, self.disparities[self.cnt], self.disparities[self.cnt + 1], dmatches)
        if len(prev_kpts) == 0 or len(curr_kpts) == 0:  # Could not track features
            self.append_kpts_match_info(prev_kpts, curr_kpts, dmatches)
            return None

        # Calculate essential matrix and the correct pose
        prev_3d_pts, curr_3d_pts = self.calc_3d(l_prev_pts, r_prev_pts, l_curr_pts, r_curr_pts)

        if self.method == "svd":
            transform_matrix = self.svd_based_translation_estimation(l_prev_pts, l_curr_pts, prev_3d_pts, curr_3d_pts)
        elif self.method == "greedy":
            transform_matrix = self.greedy_translation_estimation(l_prev_pts, l_curr_pts, prev_3d_pts, curr_3d_pts)

        self.append_kpts_match_info(prev_kpts, curr_kpts, dmatches)
        return transform_matrix

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
            prev_disps, p_mask = get_disps(prev_kpts, self.cnt)
            curr_disps, c_mask = get_disps(curr_kpts, self.cnt + 1)
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
        limg = cv2.cvtColor(self.left_imgs[cnt], cv2.COLOR_BGR2GRAY)
        rimg = cv2.cvtColor(self.right_imgs[cnt], cv2.COLOR_BGR2GRAY)
        WIN_SIZE = 3
        disps = []
        for pt in pts:
            pt_int = (int(pt[0]), int(pt[1]))
            templ = limg[pt_int[1] - WIN_SIZE:pt_int[1] + WIN_SIZE, pt_int[0] - WIN_SIZE:pt_int[0] + WIN_SIZE]
            result = cv2.matchTemplate(rimg[pt_int[1] - WIN_SIZE:pt_int[1] + WIN_SIZE, :pt_int[0]], templ, cv2.TM_SQDIFF)
            # cv2.TM_CCOEFF_NORMEDの場合は第4戻り値を使う
            _, _, loc, _ = cv2.minMaxLoc(result)
            disp = pt[0] - loc[0] - WIN_SIZE // 2       # テンプレートの中心に来るように補正
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

    def svd_based_translation_estimation(
        self,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_3d_pts: np.ndarray, curr_3d_pts: np.ndarray,
    ) -> np.ndarray:
        prev_3d_pts = np.vstack((prev_3d_pts.T, np.ones((1, prev_3d_pts.shape[0]))))
        curr_3d_pts = np.vstack((curr_3d_pts.T, np.ones((1, curr_3d_pts.shape[0]))))

        # RANSAC
        # FIXME RANSACでベストモデルを推定するよりも全特徴点を使った方が精度が良い
        # max_trial = 1000
        # min_error = 1e10
        # early_termination = 0
        # early_termination_thd = 100
        # T = None
        # for _ in range(max_trial):
        #     sample_idx = np.random.choice(range(prev_3d_pts.shape[1]), int(prev_3d_pts.shape[1] / 2))
        #     sample_prev_3d_pts = prev_3d_pts[:, sample_idx]
        #     sample_curr_3d_pts = curr_3d_pts[:, sample_idx]
        #     sample_avg_prev_3d_pts = np.mean(sample_prev_3d_pts, axis=1).reshape((4, -1))
        #     sample_avg_curr_3d_pts = np.mean(sample_curr_3d_pts, axis=1).reshape((4, -1))

        #     U, _, V = np.linalg.svd((sample_prev_3d_pts - sample_avg_prev_3d_pts) @ (sample_curr_3d_pts - sample_avg_curr_3d_pts).T)
        #     sample_R = V.T @ U.T
        #     if np.linalg.det(sample_R) < 0:
        #         continue
        #     sample_t = sample_avg_curr_3d_pts - sample_R @ sample_avg_prev_3d_pts
        #     sample_T = np.eye(4)
        #     sample_T[: 3, : 3] = sample_R[: 3, : 3].T
        #     sample_T[: 3, 3] = sample_t[: 3, 0]

        #     res = self.reprojection_residuals(sample_T, prev_pixes, curr_pixes, prev_3d_pts, curr_3d_pts)
        #     res = res.reshape((prev_3d_pts.shape[1] * 2, 2))
        #     err = np.sum(np.linalg.norm(res, axis=1))
        #     if err < min_error:
        #         min_error = err
        #         T = sample_T
        #         early_termination = 0
        #     else:
        #         early_termination += 1
        #     if early_termination == early_termination_thd:
        #         break

        avg_prev_3d_pts = np.mean(prev_3d_pts, axis=1).reshape((4, -1))
        avg_curr_3d_pts = np.mean(curr_3d_pts, axis=1).reshape((4, -1))

        U, _, V = np.linalg.svd((prev_3d_pts - avg_prev_3d_pts) @ (curr_3d_pts - avg_curr_3d_pts).T)
        R = V.T @ U.T
        if np.linalg.det(R) < 0:
            return None
        t = avg_curr_3d_pts - R @ avg_prev_3d_pts
        T = np.eye(4)
        T[: 3, : 3] = R[: 3, : 3].T
        T[: 3, 3] = t[: 3, 0]
        return T

    def greedy_translation_estimation(
        self,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_3d_pts: np.ndarray, curr_3d_pts: np.ndarray,
        max_iter: int = 100
    ) -> np.ndarray:
        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0
        early_termination_thd = 10
        for _ in range(max_iter):
            sample_idx = np.random.choice(range(prev_pixes.shape[0]), 20)

            sample_q1 = prev_pixes[sample_idx]
            sample_q2 = curr_pixes[sample_idx]
            sample_Q1 = prev_3d_pts[sample_idx]
            sample_Q2 = curr_3d_pts[sample_idx]

            # Perform least squares optimization
            in_guess = np.zeros(6)  # Make the start guess
            opt_res = least_squares(
                self.optimize_function,                        # Function to minimize
                in_guess,                                           # Initial guess
                method='lm',                                        # Levenberg-Marquardt algorithm
                max_nfev=200,                                       # Max number of function evaluations
                args=(sample_q1, sample_q2, sample_Q1, sample_Q2)   # Additional arguments to pass to the function
            )

            # Calculate the error for the optimized transformation
            res = self.optimize_function(opt_res.x, prev_pixes, curr_pixes, prev_3d_pts, curr_3d_pts)
            res = res.reshape((prev_3d_pts.shape[0] * 2, 2))
            err = np.sum(np.linalg.norm(res, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if err < min_error:
                min_error = err
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_thd:
                # If we have not fund any better result in early_termination_threshold iterations
                break
        r = out_pose[:3]             # Get the rotation vector
        R, _ = cv2.Rodrigues(r)      # Make the rotation matrix
        t = -out_pose[3:]            # Get the translation vector
        T = self._form_transf(R, t)  # Make the transformation matrix
        return T

    def optimize_function(
            self,
            dof: np.ndarray,
            p_pixes: np.ndarray, c_pixes: np.ndarray,
            p3d_pts: np.ndarray, c3d_pts: np.ndarray
    ):
        """Optimize function: Calculates the reprojection residuals from the rotation and translation vector

        Args:
            dof (np.ndarray): Rotation vector and translation vector
            prev_pixes (np.ndarray): Pixel points in image 1
            curr_pixes (np.ndarray): Pixel points in image 2
            prev_3d_pts (np.ndarray): 3D points in image 1
            curr_3d_pts (np.ndarray): 3D points in image 2

        Returns:
            np.ndarray: Reprojection residuals (Flattened)
        """
        r = dof[:3]  # Get the rotation vector
        R, _ = cv2.Rodrigues(r)  # Create the rotation matrix from the rotation vector
        t = dof[3:]  # Get the translation vector
        transf = self._form_transf(R, t)    # Create the transformation matrix from the rotation matrix and translation vector
        ones = np.ones((p_pixes.shape[0], 1))
        p3d_pts = np.hstack([p3d_pts, ones]).T
        c3d_pts = np.hstack([c3d_pts, ones]).T
        return self.reprojection_residuals(transf, p_pixes, c_pixes, p3d_pts, c3d_pts)

    def reprojection_residuals(
        self,
        transf: np.ndarray,
        prev_pixes: np.ndarray, curr_pixes: np.ndarray,
        prev_3d_pts: np.ndarray, curr_3d_pts: np.ndarray
    ) -> np.ndarray:
        """Calculate residuals for reprojection

        Args:
            transf (np.ndarray): Transformation matrix in the homogeneous form
            prev_pixes (np.ndarray): Pixel points in image 1
            curr_pixes (np.ndarray): Pixel points in image 2
            prev_3d_pts (np.ndarray): 3D points in image 1
            curr_3d_pts (np.ndarray): 3D points in image 2

        Returns:
            np.ndarray: Reprojection residuals (Flattened)
        """
        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = self.P_l @ transf
        b_projection = self.P_l @ np.linalg.inv(transf)

        q1_pred = curr_3d_pts.T @ f_projection.T        # Project 3D points from i'th image to i-1'th image
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]  # Un-homogenize
        q2_pred = prev_3d_pts.T @ b_projection.T    # Project 3D points from i-1'th image to i'th image
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]  # Un-homogenize
        residuals = np.vstack([q1_pred - prev_pixes.T, q2_pred - curr_pixes.T]).flatten()  # Calculate the residuals
        return residuals

    def append_kpts_match_info(self, prev_kpts, curr_kpts, dmatches):
        self.matched_prev_kpts.append(prev_kpts)
        self.matched_curr_kpts.append(curr_kpts)
        self.matches.append(dmatches)

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
                process_times=self.process_times[i],
                kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in kpts],
                descs=self.left_descs[i],
                disp=self.disparities[i],
                translation=self.Ts[i],
                matches=[[m.queryIdx, m.trainIdx, m.imgIdx, m.distance] for m in self.matches[i]] if self.matches[i] is not None else None,
                matched_prev_kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in self.matched_prev_kpts[i]] if self.matched_prev_kpts[i] is not None else None,
                matched_curr_kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in self.matched_curr_kpts[i]] if self.matched_curr_kpts[i] is not None else None,
            )
