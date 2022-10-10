from __future__ import annotations
from re import I
import cv2
import quaternion
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# https://github.com/niconielsen32/ComputerVision/tree/a3caf60f0134704958879b9c7e3ef74090ca6579/VisualOdometry


class VisualOdometry():
    def __init__(self, camera_params, imgs) -> None:
        self.intrinsic_l = camera_params['intrinsic']
        self.extrinsic_l = camera_params['extrinsic']
        self.P_l = camera_params['projection']
        self.K_l = self.P_l[0:3, 0:3]
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
        self.K_l = self.intrinsic_l
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
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1) /
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
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
        num_disp: int = 300, winSize: tuple = (15, 15)
    ) -> None:
        super().__init__(left_camera_params, left_imgs)

        self.intrinsic_r = right_camera_params['intrinsic']
        self.extrinsic_r = right_camera_params['extrinsic']
        self.P_r = right_camera_params['projection']
        self.K_r = self.P_r[0:3, 0:3]
        self.right_imgs = right_imgs

        self.disparity = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=10)
        # self.detector = cv2.FastFeatureDetector_create()
        # self.detector = cv2.ORB_create()
        self.detector = cv2.AKAZE_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.max_error = 6
        self.lk_params = dict(winSize=winSize, flags=cv2.MOTION_AFFINE, maxLevel=11, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.03))

        l_img, r_img = self.load_img(self.cnt)
        self.disparities = [np.divide(self.disparity.compute(l_img, r_img).astype(np.float32), 16)]
        l_kpts, l_desc = self.detector.detectAndCompute(l_img, None)

        self.left_kpts = [l_kpts]
        self.left_descs = [l_desc]
        self.matches = [None]
        self.matched_prev_kpts = [None]
        self.matched_curr_kpts = [None]

    def estimate_pose(self):
        left_curr_img, right_curr_img = self.load_img(self.cnt + 1)
        self.disparities.append(np.divide(self.disparity.compute(left_curr_img, right_curr_img).astype(np.float32), 16))
        # tp1_l, tp2_l = self.detectAndTrackFeaturesByOptFlow(l1_img, l2_img)
        tp1_l, tp2_l = self.detectAndTrackFeatures(self.cnt, left_curr_img)
        if len(tp1_l) == 0 or len(tp2_l) == 0:  # Could not track features
            return None
        tp1_l, tp2_l, pt1_l, pt1_r, pt2_l, pt2_r = self.calculate_right_qs(tp1_l, tp2_l, self.disparities[self.cnt], self.disparities[self.cnt + 1])
        # for q1, q2 in zip(tp1_l, tp2_l):
        #     cv2.line(l1_img, (int(q1[0]), int(q1[1])), (int(q2[0]), int(q2[1])), (0, 255, 0), 1)
        # cv2.imwrite(f'./keypoints_{self.cnt:05d}.png', l1_img)

        # for kp in kp1_l:
        #     cv2.circle(l1_img, (int(kp.pt[0]), int(kp.pt[1])), 2, (0, 0, 255), 1)
        # fig, ax = plt.subplots()
        # ax.imshow(self.disparities[-1])
        # fig.savefig(f'./disparity_{self.cnt:05d}.png', dpi=300, bbox_inches='tight', pad_inches=0)
        # plt.cla()
        Q1, Q2 = self.calc_3d(pt1_l, pt1_r, pt2_l, pt2_r)
        transformation_matrix = self.estimate_transform_matrix(pt1_l, pt2_l, Q1, Q2)
        self.matched_prev_kpts.append(tp1_l)
        self.matched_curr_kpts.append(tp2_l)
        self.cnt += 1
        return transformation_matrix

    def detectAndTrackFeaturesByOptFlow(self, img1: np.ndarray, img2: np.ndarray):
        kp1_l = self.get_keypoints(img1, 10, 10)
        tp1_l, tp2_l = self.track_keypoints(img1, img2, kp1_l, 6)
        return tp1_l, tp2_l

    def detectAndTrackFeatures(self, i: int, curr_img: np.ndarray):
        prev_kpts = self.left_kpts[i]
        prev_descs = self.left_descs[i]
        curr_kpts, curr_descs = self.detector.detectAndCompute(curr_img, None)
        self.left_kpts.append(curr_kpts)
        self.left_descs.append(curr_descs)
        matches = self.bf.match(prev_descs, curr_descs)

        tp1 = []
        tp2 = []
        matches = sorted(matches, key=lambda x: x.distance)
        for i in range(min(50, len(matches))):
            tp1.append(prev_kpts[matches[i].queryIdx])
            tp2.append(curr_kpts[matches[i].trainIdx])
        self.matches.append(matches[:i + 1])
        # tp1 = np.array([kp1[m.queryIdx].pt for m in matches])
        # tp2 = np.array([kp2[m.trainIdx].pt for m in matches])
        # dst = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
        # cv2.imwrite(f"./match_pt_{self.cnt:05d}.png", dst)
        # img = np.hstack((img1, img2))
        # for p1, p2 in zip(tp1, tp2):
        #     cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0] + img1.shape[1]), int(p2[1])), (0, 255, 0), 1)
        # cv2.imwrite(f"./match_pt_{self.cnt:05d}.png", img)
        return tp1, tp2

    def get_keypoints(self, img: np.ndarray, tile_h: int, tile_w: int):
        def get_kps(x, y):
            impatch = img[y:y + tile_h, x:x + tile_w]   # Get the image tile
            keypoints = self.detector.detect(impatch)  # Detect keypoints
            for pt in keypoints:    # Correct the coordinate for the point
                pt.pt = (pt.pt[0] + x, pt.pt[1] + y)
            if len(keypoints) > 10:  # Get the 10 best keypoints
                keypoints = sorted(keypoints, key=lambda x: -x.response)
                return keypoints[:10]
            return keypoints
        h, w, _ = img.shape  # Get the image height and width
        kp_list = [get_kps(x, y) for y in range(0, h, tile_h) for x in range(0, w, tile_w)]  # Get the keypoints for each of the tiles
        kp_list_flatten = np.concatenate(kp_list)   # Flatten the keypoint list
        return kp_list_flatten

    def track_keypoints(self, img1: np.ndarray, img2: np.ndarray, kp1, max_error=10):
        trackpoints1 = np.expand_dims(cv2.KeyPoint_convert(kp1), axis=1)
        trackpoints2, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, trackpoints1, None, **self.lk_params)  # Use optical flow to find tracked counterparts
        trackable = st.astype(bool)  # Convert the status vector to boolean so we can use it as a mask
        under_thresh = np.where(err[trackable] < max_error, True, False)  # Create a maks there selects the keypoints there was trackable and under the max error

        # Use the mask to select the keypoints
        trackpoints1 = trackpoints1[trackable][under_thresh]
        trackpoints2 = np.around(trackpoints2[trackable][under_thresh])

        # Remove the keypoints there is outside the image
        h, w, _ = img1.shape
        in_bounds = np.where(np.logical_and(trackpoints2[:, 1] < h, trackpoints2[:, 0] < w), True, False)
        trackpoints1 = trackpoints1[in_bounds]
        trackpoints2 = trackpoints2[in_bounds]
        return trackpoints1, trackpoints2

    def calculate_right_qs(self, q1: list[cv2.KeyPoint], q2: list[cv2.KeyPoint], disp1, disp2, min_disp=10.0, max_disp=512.0):
        def get_idxs(q: cv2.KeyPoint, disp):
            q_pts = np.array([q_.pt for q_ in q])
            q_idx = q_pts.astype(int)
            disp = disp.T[q_idx[:, 0], q_idx[:, 1]]
            return disp, np.where(np.logical_and(min_disp < disp, disp < max_disp), True, False)

        # Get the disparity's for the feature points and mask for min_disp & max_disp
        disp1, mask1 = get_idxs(q1, disp1)
        disp2, mask2 = get_idxs(q2, disp2)

        in_bounds = np.logical_and(mask1, mask2)    # Combine the masks
        kpt1_l, kpt2_l, disp1, disp2 = np.array(q1)[in_bounds], np.array(q2)[in_bounds], disp1[in_bounds], disp2[in_bounds]  # Get the feature points and disparity's there was in bounds
        self.matches[-1] = list(np.array(self.matches[-1])[in_bounds])  # Update the matches
        q1_l = np.array([q.pt for q in kpt1_l])
        q2_l = np.array([q.pt for q in kpt2_l])

        # Calculate the right feature points
        q1_r, q2_r = np.copy(q1_l), np.copy(q2_l)
        q1_r[:, 0] -= disp1
        q2_r[:, 0] -= disp2
        return list(kpt1_l), list(kpt2_l), q1_l, q1_r, q2_l, q2_r

    def estimate_transform_matrix(self, q1: np.ndarray, q2: np.ndarray, Q1: np.ndarray, Q2: np.ndarray, max_iter: int = 100):
        early_termination_threshold = 5

        # Initialize the min_error and early_termination counter
        min_error = float('inf')
        early_termination = 0

        for _ in range(max_iter):
            # Choose 6 random feature points
            sample_idx = np.random.choice(range(q1.shape[0]), 6)
            sample_q1, sample_q2, sample_Q1, sample_Q2 = q1[sample_idx], q2[sample_idx], Q1[sample_idx], Q2[sample_idx]

            # Make the start guess
            in_guess = np.zeros(6)
            # Perform least squares optimization
            opt_res = least_squares(self.reprojection_residuals, in_guess, method='lm', max_nfev=200, args=(sample_q1, sample_q2, sample_Q1, sample_Q2))

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
            if early_termination == early_termination_threshold:
                # If we have not fund any better result in early_termination_threshold iterations
                break

        r = out_pose[:3]    # Get the rotation vector
        R, _ = cv2.Rodrigues(r)  # Make the rotation matrix
        t = out_pose[3:]    # Get the translation vector
        transformation_matrix = self._form_transf(R, t)  # Make the transformation matrix
        return transformation_matrix

    def reprojection_residuals(self, dof, q1: np.ndarray, q2: np.ndarray, Q1, Q2):
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

    def calc_3d(self, q1_l: np.ndarray, q1_r: np.ndarray, q2_l: np.ndarray, q2_r: np.ndarray):
        Q1 = cv2.triangulatePoints(self.P_l, self.P_r, q1_l.T, q1_r.T)  # Triangulate points from i-1'th image
        Q1 = np.transpose(Q1[:3] / Q1[3])   # Un-homogenize
        Q2 = cv2.triangulatePoints(self.P_l, self.P_r, q2_l.T, q2_r.T)  # Triangulate points from i'th image
        Q2 = np.transpose(Q2[:3] / Q2[3])   # Un-homogenize
        return Q1, Q2

    def load_img(self, i: int) -> list[np.ndarray, np.ndarray]:
        l_img = self.left_imgs[i]
        r_img = self.right_imgs[i]
        return l_img, r_img

    def save_results(self, i: int, base_src: str = "./result"):
        kpts = self.left_kpts[i]
        np.savez(
            f"{base_src}/{i:04d}.npz",
            kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in kpts],
            descs=self.left_descs[i],
            disp=self.disparities[i],
            matches=[[m.queryIdx, m.trainIdx, m.imgIdx, m.distance] for m in self.matches[i]] if self.matches is not None else None,
            matched_prev_kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in self.matched_prev_kpts[i]] if self.matched_prev_kpts is not None else None,
            matched_curr_kpts=[[kpt.pt[0], kpt.pt[1], kpt.size, kpt.angle, kpt.response, kpt.octave, kpt.class_id] for kpt in self.matched_curr_kpts[i]] if self.matched_curr_kpts is not None else None,
        )
