from __future__ import annotations
import cv2
import os
import quaternion
import warnings
import numpy as np
from scipy.optimize import least_squares

# https://github.com/niconielsen32/ComputerVision/tree/a3caf60f0134704958879b9c7e3ef74090ca6579/VisualOdometry



class VisualOdometry():
    def __init__(self, camera_params, imgs) -> None:
        self.extrinsic_l = camera_params['extrinsic']
        self.P_l = camera_params['projection']
        self.K_l = camera_params['intrinsic']
        self.left_imgs = imgs
        self.cnt = 0

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = -t
        T[3, 3] = 1.0
        return T

    def load_img(self, i: int) -> np.ndarray:
        return self.left_imgs[i]


class MonocularVisualOdometry(VisualOdometry):
    def __init__(self, left_camera_params, left_imgs) -> None:
        super().__init__(left_camera_params, left_imgs)
        # self.P_l = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

        self.orb = cv2.ORB_create(3000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    def estimate_pose(self):
        q1, q2 = self.get_matches(self.cnt)
        transf = self.estimate_transform_matrix(q1, q2)
        self.cnt += 1
        return np.linalg.inv(transf)

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object
        Parameters
        ----------
        i (int): The current frame
        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        # Find the keypoints and descriptors with ORB
        kp1, des1 = self.orb.detectAndCompute(self.left_imgs[i], None)
        kp2, des2 = self.orb.detectAndCompute(self.left_imgs[i + 1], None)
        matches = self.flann.knnMatch(des1, des2, k=2)  # Find matches

        # Find the matches there do not have a to high distance
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        # draw_params = dict(
        #     matchColor=-1,  # draw matches in green color
        #     singlePointColor=None,
        #     matchesMask=None,  # draw only inliers
        #     flags=2
        # )

        # img3 = cv2.drawMatches(self.left_imgs[i], kp1, self.left_imgs[i+1], kp2, good, None, **draw_params)
        # cv2.imwrite("orb_matches.png", img3)
        # cv2.imshow("image", img3)
        # cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def estimate_transform_matrix(self, q1, q2):
        """
        Calculates the transformation matrix
        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        E, _ = cv2.findEssentialMat(q1, q2, cameraMatrix=self.K_l, threshold=1)
        R, t = self.decomp_essential_mat(E, q1, q2)
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

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


class StereoVisualOdometry(VisualOdometry):
    def __init__(
        self,
        left_camera_params, right_camera_params, left_imgs, right_imgs,
        detector, descriptor,
        num_disp: int = 300, winSize: tuple = (15, 15),
        base_rot: np.ndarray = np.eye(3)
    ) -> None:
        super().__init__(left_camera_params, left_imgs)

        self.extrinsic_r = right_camera_params['extrinsic']
        self.P_r = right_camera_params['projection']
        self.K_r = right_camera_params['intrinsic']
        self.right_imgs = right_imgs

        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=10)
        self.detector = detector
        self.descriptor = descriptor
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.max_error = 6
        self.base_rot = base_rot

        self.lk_params = dict(winSize=winSize, flags=cv2.MOTION_AFFINE, maxLevel=11, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        l_img, r_img = self.load_img(self.cnt)
        l_kpts = self.detector.detect(l_img, None)
        l_kpts, l_descs = self.descriptor.compute(l_img, l_kpts)
        l_descs = np.array(l_descs, dtype=np.uint8)
        self.disparities = [np.divide(self.disparity.compute(l_img, r_img).astype(np.float32), 16)]

        self.left_kpts = [l_kpts]
        self.left_descs = [l_descs]
        self.matches = [None]
        self.matched_prev_kpts = [None]
        self.matched_curr_kpts = [None]

    def estimate_pose(self):
        # Load images
        left_curr_img, right_curr_img = self.load_img(self.cnt + 1)

        # Calculate disparity
        self.disparities.append(np.divide(self.disparity.compute(left_curr_img, right_curr_img).astype(np.float32), 16))

        # Detect and track keypoints
        tp1_l, tp2_l, matches = self.detectAndTrackFeatures(self.cnt, left_curr_img)
        if len(tp1_l) == 0 or len(tp2_l) == 0:  # Could not track features
            warnings.warn("Cannot track features")
            self.matched_prev_kpts.append(tp1_l)
            self.matched_curr_kpts.append(tp2_l)
            self.matches.append(matches)
            self.cnt += 1
            return None

        # Find the corresponding points in the right image
        tp1_l, tp2_l, matches, pt1_l, pt1_r, pt2_l, pt2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[self.cnt], self.disparities[self.cnt + 1], matches)

        # Delete keypoints if needed
        new_tp1_l, new_tp2_l, new_matches, new_pt1_l, new_pt2_l, new_pt1_r, new_pt2_r = [], [], [], [], [], [], []
        cnt = 0
        for t1, t2, match, p1_l, p1_r, p2_l, p2_r in zip(tp1_l, tp2_l, matches, pt1_l, pt1_r, pt2_l, pt2_r):
            if p1_l[1] > 300 or p2_l[1] > 300:
                continue
            if p1_l[1] < 100 or p2_l[1] < 100:
                continue
            if p1_l[0] < 50 or p2_l[0] < 50:
                continue
            if p1_l[0] > left_curr_img.shape[1] - 50 or p2_l[0] > left_curr_img.shape[1] - 50:
                continue
            new_tp1_l.append(t1)
            new_tp2_l.append(t2)
            new_pt1_l.append(p1_l)
            new_pt2_l.append(p2_l)
            new_pt1_r.append(p1_r)
            new_pt2_r.append(p2_r)
            new_matches.append(cv2.DMatch(cnt, cnt, match.imgIdx, match.distance))
            cnt += 1
        matches = new_matches
        tp1_l, tp2_l, pt1_l, pt2_l, pt1_r, pt2_r = np.array(new_tp1_l), np.array(new_tp2_l), np.array(new_pt1_l), np.array(new_pt2_l), np.array(new_pt1_r), np.array(new_pt2_r)

        if len(tp1_l) <= 5 or len(tp2_l) <= 5:  # Could not track features
            warnings.warn("Cannot track features")
            self.matched_prev_kpts.append(tp1_l)
            self.matched_curr_kpts.append(tp2_l)
            self.matches.append(matches)
            self.cnt += 1
            return None

        # Calculate essential matrix and the correct pose
        Q1, Q2 = self.calc_3d(pt1_l, pt1_r, pt2_l, pt2_r)
        transformation_matrix = self.estimate_transform_matrix(pt1_l, pt2_l, Q1, Q2)

        self.matched_prev_kpts.append(tp1_l)
        self.matched_curr_kpts.append(tp2_l)
        self.matches.append(matches)
        self.cnt += 1
        return transformation_matrix

    def detectAndTrackFeatures(self, i: int, curr_img: np.ndarray):
        prev_kpts = self.left_kpts[i]
        prev_descs = self.left_descs[i]
        curr_kpts = self.detector.detect(curr_img, None)
        curr_kpts, curr_descs = self.descriptor.compute(curr_img, curr_kpts)
        curr_descs = np.array(curr_descs, dtype=np.uint8)
        self.left_kpts.append(curr_kpts)
        self.left_descs.append(curr_descs)
        matches = self.bf.match(prev_descs, curr_descs)

        tp1, tp2, new_matches = [], [], []
        matches = sorted(matches, key=lambda x: x.distance)
        for i in range(min(50, len(matches))):
            tp1.append(prev_kpts[matches[i].queryIdx])
            tp2.append(curr_kpts[matches[i].trainIdx])
            new_matches.append(cv2.DMatch(i, i, matches[i].imgIdx, matches[i].distance))
        return tp1, tp2, new_matches

    def calculate_right_qs(self, q1: list[cv2.KeyPoint], q2: list[cv2.KeyPoint], disps1: np.ndarray, disps2: np.ndarray, matches: list[cv2.DMatch], min_disp: float = 10.0, max_disp: float = 512.0):
        """Find correspond points in the right image and returns keypoints and descriptors in left and right image

        Args:
            q1 (list[cv2.KeyPoint]): List of keypoints in the first image
            q2 (list[cv2.KeyPoint]): List of keypoints in the second image
            disps1 (np.ndarray): Keypoint descripoints in the first image
            disps2 (np.ndarray): Keypoint descripoints in the second image
            matches (list[cv2.DMatch]): _description_
            min_disp (float, optional): _description_. Defaults to 10.0.
            max_disp (float, optional): _description_. Defaults to 512.0.

        Returns:
            list[cv2.KeyPoint], list[cv2.KeyPoint], list[cv2.DMatch], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]: List of keypoints, descriptors and masks in left and right image
        """

        def get_idxs(q: list, disp):
            q_pts = np.array([q_.pt for q_ in q])
            q_idx = q_pts.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)

        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disps1, mask1 = get_idxs(q1, disps1)
        disps2, mask2 = get_idxs(q2, disps2)

        masks = np.logical_and(mask1, mask2)    # Combine the masks
        kpt1_l, kpt2_l, disps1_masked, disps2_masked, matches_masked = [], [], [], [], []
        mask_cnt = 0
        for mask, kpt1, kpt2, disp1, disp2, match in zip(masks, q1, q2, disps1, disps2, matches):
            if mask:
                kpt1_l.append(kpt1)
                kpt2_l.append(kpt2)
                disps1_masked.append(disp1)
                disps2_masked.append(disp2)
                matches_masked.append(cv2.DMatch(mask_cnt, mask_cnt, match.imgIdx, match.distance))
                mask_cnt += 1
        q1_l = np.array([q.pt for q in kpt1_l])
        q2_l = np.array([q.pt for q in kpt2_l])

        # Calculate the right feature points
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disps1_masked
        q2_r[:, 0] -= disps2_masked
        return list(kpt1_l), list(kpt2_l), matches_masked, q1_l, q1_r, q2_l, q2_r

    def calc_3d(self, q1_l: np.ndarray, q1_r: np.ndarray, q2_l: np.ndarray, q2_r: np.ndarray) -> list[np.ndarray]:
        """Calculate 3D position from correspoind points in left and right image

        Args:
            q1_l (np.ndarray): _description_
            q1_r (np.ndarray): _description_
            q2_l (np.ndarray): _description_
            q2_r (np.ndarray): _description_

        Returns:
            list: List of 3D points
        """
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)  # Triangulate points from i-1'th image
        Q1 = np.transpose(Q1[:3] / Q1[3])   # Un-homogenize
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)  # Triangulate points from i'th image
        Q2 = np.transpose(Q2[:3] / Q2[3])   # Un-homogenize
        return Q1, Q2

    def estimate_transform_matrix(self, q1: np.ndarray, q2: np.ndarray, Q1: np.ndarray, Q2: np.ndarray, max_iter: int = 100):
        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0
        early_termination_thd = 5

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            in_guess = np.zeros(6)  # Make the start guess
            opt_res = least_squares(
                self.reprojection_residuals,
                in_guess,
                method='lm',
                max_nfev=200,
                args=(sample_q1, sample_q2, sample_Q1, sample_Q2)
            )  # Perform least squares optimization

            # Calculate the error for the optimized transformation
            error = self.reprojection_residuals(opt_res.x, q1, q2, Q1, Q2)
            error = error.reshape((Q1.shape[0] * 2, 2))
            error = np.sum(np.linalg.norm(error, axis=1))

            # Check if the error is less the the current min error. Save the result if it is
            if error < min_error:
                min_error = error
                out_pose = opt_res.x
                early_termination = 0
            else:
                early_termination += 1
            if early_termination == early_termination_thd:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        r = out_pose[:3]    # Get the rotation vector
        R, _ = cv2.Rodrigues(r)  # Make the rotation matrix
        t = out_pose[3:]    # Get the translation vector
        t = self.base_rot @ t
        transformation_matrix = self._form_transf(R, t)  # Make the transformation matrix
        return transformation_matrix

    def reprojection_residuals(self, dof: np.ndarray, q1: np.ndarray, q2: np.ndarray, Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
        """Calculate residuals for reprojection

        Args:
            dof (np.ndarray): Transformation matrix
            q1 (np.ndarray): Pixel points in image 1
            q2 (np.ndarray): Pixel points in image 2
            Q1 (np.ndarray): 3D points in image 1
            Q2 (np.ndarray): 3D points in image 2

        Returns:
            np.ndarray: Reprojection residuals (Flattened)
        """

        r = dof[:3]  # Get the rotation vector
        R, _ = cv2.Rodrigues(r)  # Create the rotation matrix from the rotation vector
        t = dof[3:]  # Get the translation vector
        transf = self._form_transf(R, t)    # Create the transformation matrix from the rotation matrix and translation vector

        # Create the projection matrix for the i-1'th image and i'th image
        f_projection = np.matmul(self.P_l, transf)
        b_projection = np.matmul(self.P_l, np.linalg.inv(transf))

        # Make the 3D points homogenize
        ones = np.ones((q1.shape[0], 1))
        Q1 = np.hstack([Q1, ones])
        Q2 = np.hstack([Q2, ones])

        q1_pred = Q2.dot(f_projection.T)        # Project 3D points from i'th image to i-1'th image
        q1_pred = q1_pred[:, :2].T / q1_pred[:, 2]  # Un-homogenize
        q2_pred = Q1.dot(b_projection.T)    # Project 3D points from i-1'th image to i'th image
        q2_pred = q2_pred[:, :2].T / q2_pred[:, 2]  # Un-homogenize
        residuals = np.vstack([q1_pred - q1.T, q2_pred - q2.T]).flatten()  # Calculate the residuals
        return residuals

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

    def save_results(self, last_img_idx: int, step: int, base_src: str = "./result"):
        """Save VO results (Keypoints, Descriptors, Disparity, DMatches, Matched keypoints in previous image, Matched keypoints in current image)

        Args:
            last_img_idx (int): Last image index
            step (int): VO execution step
            base_src (str, optional): Directory to be stored. Defaults to "./result".
        """
        os.makedirs(base_src, exist_ok=True)
        for i, img_idx in enumerate(range(0, last_img_idx, step)):
            kpts = self.left_kpts[i]
            np.savez(
                f"{base_src}/{img_idx:04d}.npz",
                kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in kpts],
                descs=self.left_descs[i],
                disp=self.disparities[i],
                matches=[[m.queryIdx, m.trainIdx, m.imgIdx, m.distance] for m in self.matches[i]] if self.matches[i] is not None else None,
                matched_prev_kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in self.matched_prev_kpts[i]] if self.matched_prev_kpts[i] is not None else None,
                matched_curr_kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in self.matched_curr_kpts[i]] if self.matched_curr_kpts[i] is not None else None,
            )
